from importlib.metadata import files
import matplotlib
import numpy as np
import random
from numpy.linalg import norm
from math import sqrt, pow, ceil

from constants import *

class Particle:
    '''
    class, which will represent a moving particle
    '''
    def __init__(self, pos, vel, acc):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.diffusion_pos = np.zeros(3)

        self.kin_energy = 0     # 0.5 * M * norm(vel) ** 2
        self.pot_energy = 0

    def move(self):
        self.pos += self.vel * dt + 0.5 * self.acc * (dt ** 2)
        check_boundary(self)

    def diffusion_move(self, ):
        self.diffusion_pos += self.vel * dt + 0.5 * self.acc * (dt ** 2)

def N_grid(n):
    '''Size of the grid in amount of particles for one side'''
    return ceil(pow(n, 1 / 3))

def initialize_system(on_grid=False, sigma_for_velocity=0.5, device='CPU'):
    '''
    initializes coordinates and velocities of particles
    '''
    particles = []
    for i in range(N):
        vel = np.zeros(3)
        acc = np.zeros(3)
        pos = np.zeros(3)
        if on_grid:
            n_grid = N_grid(N)
            d = L / N_grid(N)
            vel = np.zeros(3)
            acc = np.zeros(3)
            rnd_vec = np.array([np.random.uniform(-d / 20, d / 20), np.random.uniform(-d / 20, d / 20), np.random.uniform(-d / 20, d / 20)])
            x = i % n_grid
            z = i // n_grid ** 2
            y = i // n_grid - n_grid * z
            pos = d * np.array([x, y, z]) + rnd_vec
            for k in range(3):
                vel[k] = random.normalvariate(0, sigma_for_velocity)
        elif not on_grid:
            for k in range(3):
                pos[k] = np.random.uniform(0, L)
                vel[k] = random.normalvariate(0, 0)
        particles.append(Particle(pos, vel, acc))
    # calculation of initialized accelerations and
    # subsrtacting speed of center of mass

    Vc = calculate_com_vel(particles)
    for p in particles:
        p.vel -= Vc
    for i in range(N):
        for j in range(i + 1, N):
            calculate_acceleration(particles[i], particles[j])

    return particles

def force(r):
    '''
    r is a vector from one particle to another
    '''
    d = norm(r)
    f = 4 * (12 * pow(d, -13) - 6 * pow(d, -7)) * (r / d)
    return f

def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def calculate_acceleration(part1, part2):
    r = part1.pos - part2.pos       # r_1 - r_2
    # Boundary condition realisation:
    for i in range(3):
        if abs(r[i]) > L / 2:
            r[i] = r[i] - L * sgn(r[i])
            
    dist = norm(r)
    if dist < r_cut:
        part1.acc += force(r) / M       # we add the force from only one particle acting on another to the total acc
        part2.acc -= force(r) / M
        # potential of two particle interaction, we need to add it to the total pot of one: Both have a half of the total
        part1.pot_energy += -4 * (pow(dist, -6) - pow(dist, -12))
        part2.pot_energy += 0

def check_boundary(particle):
    '''
    checks boundary conditions
    '''
    for i in range(3):
        if particle.pos[i] > L:
            particle.pos[i] -=  L
        elif particle.pos[i] < 0:
            particle.pos[i] += L

def achieve_velocities(particles):
    '''
    particles : array of N particles
    velocities : 2d array of 1d arrays-vectors of velocity at the final moment for every particle
    '''
    velocities = particles[0].vel
    for p_number in range(1, N):
        velocities = np.vstack(
            (velocities, particles[p_number].vel)
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

def calculate_com_vel(particles):
    Vc = np.zeros(3)
    for p in particles:
        for i in range(3):
            Vc[i] += p.vel[i]
    Vc /= N
    return Vc

# Mostly for plotting: МНК строится в предположении, что шумы распределены нормально из принципа максимального правдоподобия
def calculate_k(x, y, through_zero=False):
    '''
    Вычисление коэффициентов для аппроксимации зависимостью y = kx + b
    Возвращает: [k, s_k, b, s_b]
    '''
    n = len(x)
    m_x = x.mean()
    m_y = y.mean()
    m_xx = (x * x). mean()
    m_yy = (y * y).mean()
    m_xy = (x * y).mean()
    m_x_m_x = x.mean() * x.mean()
    m_y_m_y = y.mean() * y.mean()

    k = 0
    s_k = 0
    b = 0
    s_b = 0

    if through_zero:
        k = m_xy / m_xx
        s_k = (1 / sqrt(n)) * sqrt((m_yy / m_xx) - k ** 2)
        return [k, s_k]

    else:
        k = (m_xy - m_x * m_y) / (m_xx - m_x_m_x)
        b = m_y - k * m_x

        s_k = (1 / sqrt(n)) * sqrt((m_yy - m_y_m_y) / (m_xx - m_x_m_x) - k ** 2)
        s_b = s_k * sqrt(m_xx - m_x_m_x)
        return [k, s_k, b, s_b]