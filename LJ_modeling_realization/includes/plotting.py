import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from math import sqrt, pow, ceil
# import modin.pandas as pd
import pandas as pd
import csv

from constants import *
from calculations import *
# from gpu_calculations import *

#----------File stuff------------------
f_traj = open('trajectories.xyz', 'w+')
FILES = [f_traj]
for file in FILES:
    file.truncate(0)

def write_first_rows_in_files():
    for file in FILES:
        file.write(str(N) + '\n')
        file.write('\n')

def write_into_the_files(p, device='CPU'):
    if device=='CPU':
        for file in FILES:
            file.write('1 ' + str(p.pos[0]) + ' ' + str(p.pos[1]) + ' ' + str(p.pos[2]) + '\n')
    else:
        for file in FILES:
            v = p.pos.cpu().numpy()
            file.write('1 ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')

# ------------------------------------

def plot_all_energies(energies, kin, pot, show=True, output_path='./energies.csv'):
    time = np.arange(0, len(energies) * dt, dt)
    if show:
        plt.plot(time, energies, color='blue')
        plt.plot(time, kin, color='red')
        plt.plot(time, pot, color='green')
        plt.show()
    pd.DataFrame(
        {'time': time, 'Total': energies, 'Kin': kin, 'Pot': pot}
    ).to_csv(output_path, index=False)

def plot_total_energy(energies):
    time = np.arange(0, len(energies) * dt, dt)
    plt.plot(time, energies, color='blue')
    plt.show()

def get_start_hist_param(norm_vels, vels_x, vels_y, vels_z, kT, bin_number=50):
    '''
    The most important things here are starting edges, which are fixed for
    the whole averaging process,
    returning left edges for plotting a bar and all edges for calculating heights on the next steps
    '''

    heights_norm, edges_norm = np.histogram(norm_vels, bins=bin_number)
    heights_x, edges_x = np.histogram(vels_x, bins=bin_number)
    heights_y, edges_y = np.histogram(vels_y, bins=bin_number)
    heights_z, edges_z = np.histogram(vels_z, bins=bin_number)
    # readjusting edges for our Temperature, this is the edges, NOT the left edges for plotting:
    big_enough_vel = 3 * sqrt(3 * kT)
    two_thirds_of_big_enough_vel = 2 * sqrt(3 * kT)
    edges_norm = np.linspace(
        0, big_enough_vel, len(edges_norm)
    )
    edges_x = np.linspace(
        -two_thirds_of_big_enough_vel, two_thirds_of_big_enough_vel, len(edges_x)
    )
    edges_y = np.linspace(
        -two_thirds_of_big_enough_vel, two_thirds_of_big_enough_vel, len(edges_y)
    )
    edges_z = np.linspace(
        -two_thirds_of_big_enough_vel, two_thirds_of_big_enough_vel, len(edges_z)
    )
    # we take only left edges here and then we'll use width
    return {
        'norm': (edges_norm, np.array(heights_norm)),
        'x': (edges_x, np.array(heights_x)),
        'y': (edges_y, np.array(heights_y)),
        'z': (edges_z, np.array(heights_z))
    }

def get_hist_param(norm_vels, vels_x, vels_y, vels_z, edges_norm, edges_x, edges_y, edges_z):
    heights_norm, useless = np.histogram(norm_vels, bins=edges_norm)
    heights_x, useless = np.histogram(vels_x, bins=edges_x)
    heights_y, useless = np.histogram(vels_y, bins=edges_y)
    heights_z, useless = np.histogram(vels_z, bins=edges_z)

    return {
        'norm': np.array(heights_norm),
        'x': np.array(heights_x),
        'y': np.array(heights_y),
        'z': np.array(heights_z)
    }

def new_hist_plot(heights, edges, kT_avg, output_path='./histograms.csv', show=True):
    names = ['$V$', '$V_x$', '$V_y$', '$V_z$']
    dict_for_df = {}
    sb = None
    for i in range(len(heights)):
        sb = plt.subplot(2, 2, 1 + i)
        width = 0.95 * (edges[i][1] - edges[i][0])
        if i == 0:
            x = np.linspace(0, max(edges[0]), 1000)
            plt.plot(
                x,
                0.1 * (1 / pow(2 * np.pi * kT_avg , 1.5)) * 4 * np.pi * (x ** 2) * np.exp( (-(x ** 2)) / (2 * kT_avg) ), color = 'red',
            )
            plt.bar(edges[i][:-1], heights[i], width, label=f'$kT={round(kT_avg, 3)} \epsilon$')
            plt.legend(loc='best', fontsize=12)
        else:
            plt.bar(edges[i][:-1], heights[i], width)
        plt.ylabel('Процент частиц', fontsize=14)
        plt.xlabel(names[i], fontsize=14)

        x_name = names[i][1:-1] + '_heights'
        y_name = y_name = names[i][1:-1] + '_edg'
        dict_for_df[x_name] = heights[i]
        dict_for_df[y_name] = edges[i][:-1]
        
    if show:
        plt.show()
    # writing into the file: (I am writing left edges into the file)
    dict_for_df['kT_average'] = [kT_avg] * len(heights[0])
    df = pd.DataFrame(dict_for_df)
    df.to_csv(output_path, index=False)

def plot_gauss_lines(heights, edges, output_path='./gauss_lines.csv', show=True):
    names= ['$V_x$', '$V_y$', '$V_z$']
    dict_for_df = {}
    sp = None
    for i in range(len(heights)):
        sb = plt.subplot(2, 2, i + 1)
        x = np.array(edges[i][:-1]) ** 2
        y = np.log(heights[i])
        plt.scatter(x, y)
        plt.xlabel(names[i] + '$^2$')
        plt.ylabel('$ln($% частиц)')
        plt.title('Линеаризация распределения по ' + names[i])

        y_name = 'log_' + names[i][1:-1] + '_heights'
        x_name = names[i][1:-1] + '_edg_square'
        dict_for_df[y_name] = y
        dict_for_df[x_name] = x
    if show:
        plt.show()
    df = pd.DataFrame(dict_for_df)
    df.to_csv(output_path, index=False)

#--diffusion:--

def write_step_of_diffusion_and_create_writer(diffusion_step, path='diffusion.csv'):
    fieldnames = []
    fieldnames.append('t')
    for i in range(N):
        fieldnames.append(str(i) + 'x')
        fieldnames.append(str(i) + 'y')
        fieldnames.append(str(i) + 'z')
    f = open(path, 'w')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    f.write('diffusion_step: ' + str(diffusion_step ) + '\n')
    f.write('Dt: ' + str(dt * diffusion_step) + '\n')
    f.write('Dt is for diffusion' + '\n')
    writer.writeheader()
    return writer

def write_diffusion(writer: csv.DictWriter, particles, time):
    writing_dict = {}
    writing_dict['t'] = round(time, 6)
    for i in range(N):
        pos = particles[i].diffusion_pos
        writing_dict[str(i) + 'x'] = pos[0]
        writing_dict[str(i) + 'y'] = pos[1]
        writing_dict[str(i) + 'z'] = pos[2]
    writer.writerow(writing_dict)

def create_coords_and_forces_writer(coords_path = 'coords.csv', forces_path = 'forces.csv', vel_path="velocities.csv"):
    '''

    returns: (coord_writer, force_writer, vel_writer)

    '''
    fieldnames = []
    fieldnames.append('t')
    for i in range(N):
        fieldnames.append(str(i) + 'x')
        fieldnames.append(str(i) + 'y')
        fieldnames.append(str(i) + 'z')
    f = open(coords_path, 'w+')
    coord_writer = csv.DictWriter(f, fieldnames=fieldnames)
    coord_writer.writeheader()
    #
    fieldnames = []
    fieldnames.append('t')
    for i in range(N):
        fieldnames.append(str(i) + 'f_x')
        fieldnames.append(str(i) + 'f_y')
        fieldnames.append(str(i) + 'f_z')
    f = open(forces_path, 'w+')
    force_writer = csv.DictWriter(f, fieldnames=fieldnames)
    force_writer.writeheader()
    #
    fieldnames = []
    fieldnames.append('t')
    for i in range(N):
        fieldnames.append(str(i) + 'v_x')
        fieldnames.append(str(i) + 'v_y')
        fieldnames.append(str(i) + 'v_z')
    f = open(vel_path, 'w+')
    vel_writer = csv.DictWriter(f, fieldnames=fieldnames)
    vel_writer.writeheader()

    return coord_writer, force_writer, vel_writer

def write_coords_and_forces(particles, time, coord_writer, force_writer, vel_writer):
    writing_dict = {}
    writing_dict['t'] = round(time, 6)
    for i in range(N):
        pos = particles[i].pos
        image = particles[i].image
        # image = 0
        abs_pos = pos + image * L
        writing_dict[str(i) + 'x'] = abs_pos[0]
        writing_dict[str(i) + 'y'] = abs_pos[1]
        writing_dict[str(i) + 'z'] = abs_pos[2]
    coord_writer.writerow(writing_dict)

    writing_dict = {}
    writing_dict['t'] = round(time, 6)
    for i in range(N):
        acc = particles[i].acc
        writing_dict[str(i) + 'f_x'] = acc[0]
        writing_dict[str(i) + 'f_y'] = acc[1]
        writing_dict[str(i) + 'f_z'] = acc[2]
    force_writer.writerow(writing_dict)

    writing_dict = {}
    writing_dict['t'] = round(time, 6)
    for i in range(N):
        vel = particles[i].vel
        writing_dict[str(i) + 'v_x'] = vel[0]
        writing_dict[str(i) + 'v_y'] = vel[1]
        writing_dict[str(i) + 'v_z'] = vel[2]
    vel_writer.writerow(writing_dict)

# короче надо сделать просто csv, где в каждый момент времени мы записиваем для всех частиц их координаты, а оттуда уже
# можно будет достать все что нужно.