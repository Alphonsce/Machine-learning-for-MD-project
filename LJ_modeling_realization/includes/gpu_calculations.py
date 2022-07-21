from importlib.metadata import files
import numpy as np
import random
from numpy.linalg import norm
from math import sqrt, pow, ceil
import torch

from constants import *
from calculations import N_grid, sgn, calculate_com_vel, check_boundary

class Particle:
    '''
    class, which will represent a moving particle
    '''
    def __init__(self, pos, vel, acc, device='CPU'):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        if device == 'GPU':
            self.diffusion_delta_pos = torch.zeros(3, dtype=torch.float).cuda()
        else:
            self.diffusion_delta_pos = np.zeros(3)

        self.kin_energy = 0     # 0.5 * M * norm(vel) ** 2
        self.pot_energy = 0

    def move(self):
        self.pos += self.vel * dt + 0.5 * self.acc * (dt ** 2)
        check_boundary(self)

    def diffusion_move(self):
        self.diffusion_delta_pos += self.vel * dt + 0.5 * self.acc * (dt ** 2)

def initialize_system(on_grid=False, sigma_for_velocity=0.5, device='CPU'):
    '''
    initializes coordinates and velocities of particles
    '''
    particles = []
    if device == 'CPU':
        for i in range(N):
            vel = np.zeros(3)
            acc = np.zeros(3)
            pos = np.zeros(3)
            if on_grid:
                n_grid = N_grid(N)
                d = L / N_grid(N)
                vel = np.zeros(3)
                acc = np.zeros(3)
                x = d * (i % n_grid) + np.random.uniform(-d / 20, d / 20)
                y = d * (i // n_grid) + np.random.uniform(-d / 20, d / 20)
                z = d * (i // n_grid **  2) + np.random.uniform(-d / 20, d / 20)
                pos = np.array([x, y, z])
                for k in range(3):
                    vel[k] = random.normalvariate(0, sigma_for_velocity)
            elif not on_grid:
                for k in range(3):
                    pos[k] = np.random.uniform(0, L)
                    vel[k] = random.normalvariate(0, 0)
            particles.append(Particle(pos, vel, acc, device))
        # calculation of initialized accelerations and
        # subsrtacting speed of center of mass

        Vc = calculate_com_vel(particles)
        for p in particles:
            p.vel -= Vc
        for i in range(N):
            for j in range(i + 1, N):
                calculate_acceleration(particles[i], particles[j])
    #--GPU--
    elif device == 'GPU':
        print('ОПА')
        for i in range(N):
            vel = torch.zeros(3, dtype=torch.float).cuda()
            acc = torch.zeros(3, dtype=torch.float).cuda()
            pos = torch.zeros(3, dtype=torch.float).cuda()
            if on_grid:
                n_grid = N_grid(N)
                d = L / N_grid(N)
                vel = torch.zeros(3, dtype=torch.float).cuda()
                acc = torch.zeros(3, dtype=torch.float).cuda()
                x = d * (i % n_grid) + np.random.uniform(-d / 20, d / 20)
                y = d * (i // n_grid) + np.random.uniform(-d / 20, d / 20)
                z = d * (i // n_grid **  2) + np.random.uniform(-d / 20, d / 20)
                pos = torch.tensor([x, y, z], dtype=torch.float).cuda()
                for k in range(3):
                    vel[k] = random.normalvariate(0, sigma_for_velocity)
            elif not on_grid:
                for k in range(3):
                    pos[k] = np.random.uniform(0, L)
                    vel[k] = random.normalvariate(0, 0)
            particles.append(Particle(pos, vel, acc, device))

        Vc = torch.from_numpy(calculate_com_vel(particles)).cuda()
        for p in particles:
            p.vel -= Vc
        for i in range(N):
            for j in range(i + 1, N):
                calculate_acceleration(particles[i], particles[j], device='GPU')
        
    return particles

def force(r, device='CPU'):
    '''
    r is a vector from one particle to another
    '''
    if device == 'CPU':
        d = norm(r)
    else:
        d = norm(r.cpu().numpy())
    f = 4 * (12 * pow(d, -13) - 6 * pow(d, -7)) * (r / d)
    return f

def calculate_acceleration(part1, part2, device='CPU'):
    r = part1.pos - part2.pos       # r_1 - r_2
    # Boundary condition realisation:
    for i in range(3):
        if abs(r[i].item()) > L / 2:
            r[i] = r[i].item() - L * sgn(r[i].item())
    if device == 'CPU':   
        dist = norm(r)
    else:
        dist = norm(r.cpu().numpy())
    if dist < r_cut:
        part1.acc += force(r, device) / M       # we add the force from only one particle acting on another to the total acc
        part2.acc -= force(r, device) / M
        # potential of two particle interaction, we need to add it to the total pot of one: Both have a half of the total
        part1.pot_energy += -4 * (pow(dist, -6) - pow(dist, -12))
        part2.pot_energy += 0

def achieve_velocities(particles, device='CPU'):
    '''
    particles : array of N particles
    velocities : 2d array of 1d arrays-vectors of velocity at the final moment for every particle
    '''
    if device == 'CPU':
        velocities = particles[0].vel
        for p_number in range(1, N):
            velocities = np.vstack(
                (velocities, particles[p_number].vel)
            )
    elif device == 'GPU':
        velocities = particles[0].vel.cpu().numpy()
        for p_number in range(1, N):
            velocities = np.vstack(
                (velocities, particles[p_number].vel.cpu().numpy())
            )
    
    vel_norms = np.array(list(map(lambda x: norm(x), velocities)))

    velocities_x = np.array([])
    velocities_y = np.array([])
    velocities_z = np.array([])
    for v in velocities:
        velocities_x = np.append(velocities_x ,v[0])
        velocities_y = np.append(velocities_y ,v[1])
        velocities_z = np.append(velocities_z ,v[2])

    return [vel_norms, velocities_x, velocities_y, velocities_z]