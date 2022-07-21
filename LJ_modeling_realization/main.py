import sys
sys.path.append('./LJ_modeling_realization/includes')
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm
from math import sqrt, pow, ceil

# import ray
# ray.init()

from includes.constants import *
from includes.calculations import *
# from includes.gpu_calculations import *
from includes.plotting import *

np.random.seed(42)

# Because temperature is an average kinetic energy of CHAOTIC movement, I'll need to substract
# the speed of center of mass from the speed of every atom to calculate the temperature


def main_cycle(spawn_on_grid=True, sigma_for_vel=0.5, verbose=1, bins_num=50, averaging_part=0.8, diffusion_step=1, device='CPU', rho_coef=2):
    '''

    main cycle, all the movements and calculations will happen here
    verbose: % of program finished to print
    diffusion_step: once every diffusion_step all coordinates will be written for diffusion plotting
    rho = 1 / rho_coef ** 3
    
    '''
    particles = initialize_system(on_grid=spawn_on_grid, sigma_for_velocity=sigma_for_vel, device=device)
    total_pot = 0
    total_kin = 0
    #---
    energies = np.array([])
    kins = np.array([])
    pots = np.array([])
    #---
    coord_writer, force_writer = create_coords_and_forces_writer(coords_path='coords3.csv', forces_path='forces3.csv')
    steps_of_averaging = int(averaging_part * TIME_STEPS)
    #---
    for ts in range(TIME_STEPS):
        write_first_rows_in_files()
        total_pot = 0
        total_kin = 0
        #-----moving---------
        for p in particles:
            p.move()
            p.kin_energy = 0.5 * norm(p.vel) ** 2
            write_into_the_files(p)
            p.vel = p.vel + 0.5 * p.acc * dt # adding 1/2 * a(t) * dt
            p.acc = np.zeros(3)
            p.pot_energy = 0
        for i in range(N):
            for j in range(i + 1, N):
                calculate_acceleration(particles[i], particles[j])
        for p in particles:
            total_kin += p.kin_energy
            total_pot += p.pot_energy
            p.vel += 0.5 * p.acc * dt   # adding 1/2 * a(t + dt)
        #---

        # energies = np.append(energies, total_kin + total_pot)
        # kins = np.append(kins, total_kin)
        # pots = np.append(pots, total_pot)      
        T_current = (2 / 3) * (total_kin) / N

        # Starting things for a set conditions:
        if ts >= TIME_STEPS - steps_of_averaging:

            write_coords_and_forces(particles=particles, time=ts * dt, coord_writer=coord_writer, force_writer=force_writer)

        #--------
        if int((0.01 * verbose * TIME_STEPS)) != 0:
            if ts % int((0.01 * verbose * TIME_STEPS)) == 0:
                print(f'{ts} steps passed, T_current = {T_current}')
        else:
            print(f'{ts} steps passed, T_current = {T_current}')

# ---------------------------------------- #
if __name__ == '__main__':
    main_cycle(spawn_on_grid=True, sigma_for_vel=1.5, bins_num=170, averaging_part=0.95, diffusion_step=50)
