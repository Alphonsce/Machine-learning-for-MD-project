import sys
from tkinter import EXCEPTION
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


def main_cycle(spawn_on_grid=True, sigma_for_vel=0.5, verbose=1, bins_num=50, averaging_part=0.8, writing_step=1, device='CPU',
    coords_path='coords.csv', forces_path='forces.csv',
    boundary_conditions=True, boundary_conditions_teleportation=True, velocity_scaler=None):
    '''

    main cycle, all the movements and calculations will happen here
    verbose: % of program finished to print
    diffusion_step: once every diffusion_step all coordinates will be written for diffusion plotting
    rho = 1 / rho_coef ** 3

    boundary_conditions: овтечает за выбор частиц из других клеток при расчете сил, ЗСЭ при этом не будет выполняться, False ставим чтобы получить F(r), тогда частицы могут бешено разогнаться

    boundary_conditions_teleportation: отвечает за 'телепортацию' частицы к противоположному краю клетки, чтобы все было норм с ЗСЭ надо оба включать

    velocity_scaler: указываем температуру к которой скейлить
    
    '''
    if str(N) not in coords_path.split('.')[0] or str(N) not in forces_path.split('.')[0]:
        raise Exception('Writing into the wrong file (amount of particles)')

    particles = initialize_system(on_grid=spawn_on_grid, sigma_for_velocity=sigma_for_vel, device=device)
    total_pot = 0
    total_kin = 0
    #---
    energies = np.array([])
    kins = np.array([])
    pots = np.array([])
    #---
    coord_writer, force_writer = create_coords_and_forces_writer(coords_path=coords_path, forces_path=forces_path)
    steps_of_averaging = int(averaging_part * TIME_STEPS)
    #---
    for ts in range(TIME_STEPS):
        write_first_rows_in_files()
        total_pot = 0
        total_kin = 0
        #-----moving---------
        for p in particles:
            p.move(boundary_conditions_teleportation)
            p.kin_energy = 0.5 * norm(p.vel) ** 2
            write_into_the_files(p)
            p.vel = p.vel + 0.5 * p.acc * dt # adding 1/2 * a(t) * dt
            p.acc = np.zeros(3)
            p.pot_energy = 0
        for i in range(N):
            for j in range(i + 1, N):
                calculate_acceleration(particles[i], particles[j], boundary_conditions)
        for p in particles:
            total_kin += p.kin_energy
            total_pot += p.pot_energy
            p.vel += 0.5 * p.acc * dt   # adding 1/2 * a(t + dt)
        #---

        # energies = np.append(energies, total_kin + total_pot)
        # kins = np.append(kins, total_kin)
        # pots = np.append(pots, total_pot)      
        T_current = (2 / 3) * (total_kin) / N

        if velocity_scaler:
            for p in particles:
                p.vel *= sqrt(velocity_scaler / T_current)

        # Starting things for a set conditions:
        if (ts >= TIME_STEPS - steps_of_averaging) and (ts % writing_step == 0):

            write_coords_and_forces(particles=particles, time=ts * dt, coord_writer=coord_writer, force_writer=force_writer)

        #--------
        if int((0.01 * verbose * TIME_STEPS)) != 0:
            if ts % int((0.01 * verbose * TIME_STEPS)) == 0:
                print(f'{ts} steps passed, T_current = {T_current}')
        else:
            print(f'{ts} steps passed, T_current = {T_current}')

# ---------------------------------------- #
if __name__ == '__main__':
    main_cycle(
        spawn_on_grid=True, sigma_for_vel=1.5, bins_num=170, averaging_part=0.95, writing_step=20,

        boundary_conditions=False,  # False, если хотим просто силы записывать
        boundary_conditions_teleportation=True,     # Если скейлер ставить и здесь True, то никто очень сильно не разгонится
        velocity_scaler=0.5,

        coords_path='coords2.csv',
        forces_path='forces2.csv'
        )

        # False, True, 0.5 - типичная настройка для записи сил


# Для каждой частицы мы берем разные 'копии' других частиц из соседних клеток, поскольку выбор оптимальных соседей для каждой частицы из главной клетки свой, 
# поэтому все что нам остается - отключить периодические условия, поскольку иначе в одной конфигурации мы можем определить силы только для одной частицы

# Но их можно отключить хитро: поскольку нас волнует только зависимость F(\vec {r}), то можно оставить телепортацию частиц при выходе из клетки, но отключить выбор частиц из других клеток при расчете сил