import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt

from numba import njit
from collections import defaultdict

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn

from LJ_modeling_realization.includes.constants import N, L

# MODE = "forces"
MODE = "movements"
# MODE = "velocities"

path = f'./dataset_objects/' + MODE + '/2_dataset_K_3.pt'        # ЗДЕСЬ БЫЛО МОДЕ ВМЕСТО movements
path_vel = f'./dataset_objects/' + "d_velocities" + '/2_dataset_K_3.pt'        # ЗДЕСЬ БЫЛО МОДЕ ВМЕСТО movements

class CFG:
    '''

    All hyperparameters are here

    '''

    N = int(path.split("/")[-1].split('_')[0])     # число атомов
    K = int(path.split("/")[-1].split('_')[-1].split('.')[0])     # можно называть это разрешением...чем число больше, тем больше размеры матрицы для атомов, фактически это число элементов в наборах p и r_cut

    L = (2 * N) ** (1 / 3) # размер одной клетки при моделировании

    r_cut = np.random.uniform(low=5, high=10, size=K).copy()
    p = np.random.uniform(low=1, high=3, size=K).copy()
    N_neig= N - 1 if N != 2 else 1

    # train_bs = 8
    # val_bs = 16
    batch_size = 1024

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    f_threshold = 5    # Если сила по какой-то координате превышает это значение, то строчка исключается, совсем маленьких по модулю сил быть не должно, если что при генерации просто r_cut поменьше надо делать
    coord_threshold = L     # Если вдруг очень большие расстояния, то надо выкидывать
    f_min_threshold = 0.05
    #
    output_size = K     # Размерность аутпута модели

naming_of_target_in_csv = {
    "forces": ["f_x", "f_y", "f_z"],
    "movements": ["s_x", "s_y", "s_z"],
    "velocities": ["v_x", "v_y", "v_z"]
}


class Descaler:
    '''
    Returns variable to the same scale
    '''
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def __call__(self, y):
        return y * (self.max - self.min) + self.min

    def scale(self, y):
        return (y - self.min) / (self.max - self.min)


def create_dataloaders(train_dataset, val_dataset, train_bs=64, val_bs=64):
    '''

    Returns train_loader, val_loader

    '''
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False)

    return train_loader, val_loader

def recieve_loaders(batch_size=64, take_one_projection_for_data=None, path=None, cut_size=None, even_for_train=False, scale_y=False, normalize_X=False, test_size=0.33):
    '''
    returns: (train_data, val_data, train_dataloader, val_dataloader, descaler)

    может создать датасет у которого всего одна проекция взята за таргет из уже существующего датасета или по пути на файл:
    take_one_projection_for_data - указываем в этом параметре номер координатной проекции (нумерация от 0)

    even_for_train - брать в качестве трейна четные, в качестве теста нечетные
    '''
    if path:
        N = int(path.split("/")[-1].split('_')[0])     # число атомов
        K = int(path.split("/")[-1].split('_')[-1].split('.')[0])     # можно называть это разрешением...чем число больше, тем больше размеры матрицы для атомов, фактически это число элементов в наборах p и r_cut

        dataset = torch.load(path)

        X = torch.vstack([elem[0] for elem in dataset])
        y = torch.vstack([elem[1] for elem in dataset])

        if normalize_X:
            normer = Normalizer()
            X = normer.fit_transform(X)
            normer = Normalizer()
            X = torch.tensor(normer.fit_transform(X))

        descaler = Descaler(1, 0)
        if scale_y:
            scaler = MinMaxScaler()
            y = scaler.fit_transform(y)
            y = torch.tensor(y, dtype=torch.float)
            descaler = Descaler(scaler.data_max_, scaler.data_min_)

        dataset = [(X[i], y[i], elem[2], elem[3]) for i, elem in enumerate(dataset)]

        if take_one_projection_for_data is not None:
            dataset = [(elem[0], elem[1][take_one_projection_for_data].unsqueeze(dim=0), elem[2], elem[3]) for elem in dataset]

        if even_for_train:
            train_data = [dataset[i] for i in range(len(dataset)) if i % 2 == 0]
            val_data = [dataset[i] for i in range(len(dataset)) if i % 2 != 0]
        else:
            train_data, val_data = train_test_split(dataset, test_size=0.33, random_state=42)
        if cut_size:
            train_data = train_data[:cut_size]
            val_data = val_data[:cut_size]
        train_dataloader, val_dataloader = create_dataloaders(train_data, val_data, train_bs=batch_size, val_bs=batch_size)

        return train_data, val_data, train_dataloader, val_dataloader, descaler

def plot_2d_result(x, y_true=None, y_pred=None, figsize=(12, 7)):
    '''
    x - is array of a_ii elements of matrix
    '''
    plt.figure(figsize=figsize)
    
    plt.xlabel('Элемент матрицы X', fontsize=20)
    plt.ylabel(f'Компонента {MODE[:-1]}', fontsize=20)

    if y_pred is not None:
        plt.scatter(x, y_pred, label='Предсказанная зависимость')
    plt.scatter(x, y_true, label='Истинная зависимость')
    plt.legend(loc='best', fontsize=20)

    plt.show()

def plot_matrix(X, Y_true, K, Y_pred=None, Y_verlet=None, figsize=(15, 15)):
    '''
    Function which plots matrix of dependencies: f_i(X_jj)
    '''
    k = len(Y_true[0])
    fig, axes = plt.subplots(k, k, figsize=figsize)

    for i in range(k):  # цикл по компонентам силы
        y_true = [elem[i] for elem in Y_true]
        if Y_pred is not None:
            y_pred = [elem[i] for elem in Y_pred]
        if Y_verlet is not None:
            y_verlet = [elem[i] for elem in Y_true] if Y_verlet else None
        for j in range(k):
            x = [elem.reshape(k, K)[j][j] for elem in X]
            axes[i][j].set_title(f'$f_{i}(X_{str(j)})$')
            axes[i][j].scatter(x, y_true, label="True", s=10)
            if Y_pred is not None:
                axes[i][j].scatter(x, y_pred, label="Predicted", s=10)
            if Y_verlet is not None:
                axes[i][j].scatter(x, y_verlet, label="True", s=10)
            axes[i][j].legend(loc="best")


def make_one_vec_transformed(vec, vec_norm, r_cut_i, p_i):
    '''
    vec: np.array - normalized vector
    norm: its norm
    r_cut_i: i-th component of
    '''
    
    # return vec_norm * vec

    # return 4 * (12 * pow(vec_norm, -13) - 6 * pow(vec_norm, -7)) * (vec)    # явный вид Леннард-Джонса
    
    # return vec / vec_norm

    return vec * np.exp(
        -np.power((vec_norm / r_cut_i), p_i)
    )
    
    # return vec * (
    #     -np.power((vec_norm / r_cut_i), p_i)        # Если вектора V_i близкие, то псевдообратная считается немного нестабильно
    #     )

    # return (pow(vec_norm, -r_cut_i) - pow(vec_norm, -p_i)) * (vec)    # Леннард-Джонс но степени - параметры

    # Если мы хотим обучаться  на скоростях и на радиус-векторах, то можно взять - расстояние, на которое 

make_matrix_transformed = np.vectorize(make_one_vec_transformed)


# -------------------
class SingleNet(nn.Module):
    '''

    Класс одиночной нейронной сети

    '''
    def __init__(self, output_size, activation=nn.ReLU(), flattened_size=CFG.K * CFG.K):
        '''
        
        FC_type: тип полносвязных слоев: 'regular' / 'simple

        convolution: сверточная часть сети

        '''
        super().__init__()

        self.FC = nn.Sequential(
        #     # nn.BatchNorm1d(flattened_size),

            nn.Linear(flattened_size, 128),
            activation,
            # nn.Dropout(0.3),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            activation,
            # nn.Dropout(0.3),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            activation,
            # nn.Dropout(0.3),
            nn.BatchNorm1d(256),

            nn.Linear(256, 256),
            activation,
            # nn.Dropout(0.3),
            nn.BatchNorm1d(256),

            # nn.Linear(256, 256),
            # activation,
            # # nn.Dropout(0.3),
            # nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            activation,
            # nn.Dropout(0.3),
            # nn.BatchNorm1d(512),
            nn.Linear(512, output_size),
        )

        # self.FC = nn.Sequential(
        #     nn.Linear(flattened_size, 64),
        #     activation,

        #     nn.Linear(64, output_size)
        # )

        # self.FC = nn.Linear(flattened_size, output_size)

    def forward(self, x):
        # x - is batch of matrices KxK

        # Здесь происходят какие-то там свертки, пуллинги и тп..

        x = self.FC(x)

        return x

def create_movements_csv(N, step=1, coords_path_to_get_movements_from=None, convert_to_csv=True, create_d_velocity=False, path_to_get_velocities_from=None):
    '''
    N - number of particles in coordsN.csv file that will be used to calculate movements
    step - step for parsing rows of dataframe

    coords_path_to_get_movements_from if not passed, will be ./coords_and_forces/coords" + str(N)

    convert_to_csv: True by default, if False - will return dataframe object

    create_d_velocity: this function is also used to create d_velocities, if True - will create d_velocitiesN.csv

    path_to_get_velocities_from: default path for velocities, to sync rows we need to select rows with a step and drop last one and then save
    '''
    if coords_path_to_get_movements_from is None:
        coord_rows = pd.read_csv("./coords_and_forces/coords" + str(N) + ".csv")[::step]
    else:
        coord_rows = pd.read_csv(coords_path_to_get_movements_from)[::step]
    coord_rows[:-1].to_csv("./coords_and_movements/coords" + str(N) + ".csv", index=False)

    if path_to_get_velocities_from is None:
        vel_rows = pd.read_csv("./coords_and_forces/velocities" + str(N) + ".csv")[::step]
    else:
        vel_rows = pd.read_csv(path_to_get_velocities_from)[::step]
    vel_rows[:-1].to_csv("./coords_and_movements/velocities" + str(N) + ".csv", index=False)

    movements = defaultdict(list)

    naming = ["t"]
    for i in range(N):
        naming.extend([str(i) + "s_x", str(i) + "s_y", str(i) + "s_z"])
    
    for row_numb in range(len(coord_rows) - 1):
        cur_coords = coord_rows.iloc[row_numb].to_numpy()
        next_coords = coord_rows.iloc[row_numb + 1].to_numpy()
        delta = (next_coords - cur_coords)
        for i in range(len(naming)):
            movements[naming[i]].append(delta[i])


    if create_d_velocity:
        d_velocities = defaultdict(list)

        naming = ["t"]
        for i in range(N):
            naming.extend([str(i) + "dv_x", str(i) + "dv_y", str(i) + "dv_z"])
        
        for row_numb in range(len(vel_rows) - 1):
            cur_coords = vel_rows.iloc[row_numb].to_numpy()
            next_coords = vel_rows.iloc[row_numb + 1].to_numpy()
            delta = (next_coords - cur_coords)
            for i in range(len(naming)):
                d_velocities[naming[i]].append(delta[i])    

        d_velocities = pd.DataFrame(d_velocities)
        d_velocities.to_csv("./coords_and_movements/d_velocities" + str(N) + ".csv", index=False)
    
    movements = pd.DataFrame(movements)
    if convert_to_csv:
        movements.to_csv("./coords_and_movements/movements" + str(N) + ".csv", index=False)
    else:
        return movements

def print_normed(V: np.array) -> None:
    print(
        V / norm(V, axis=-1)[:, np.newaxis]
    )

@njit
def _force(r):
    '''
    r is a vector from one particle to another
    '''
    d = norm(r)
    f = 4 * (12 * pow(d, -13) - 6 * pow(d, -7)) * (r / d)
    return f

calc_forces = np.vectorize(_force)

def get_rel_dists(row, atom_number, N):
    '''
    This function processes one row of csv into something that we can work with

    Returns np.array matrix that consists of relative positions vectors for passed atom_number to every other atom
    and then we can chose only closest N_neighbours in the next functions
    
    row: df.iloc[row] - typeof(row): pd.Series
    
    returns: Rel_matrix, f_vec
    '''


    s_coord = pd.Series(dtype=float)
    other_atom_numbers = [i for i in range(N) if i != atom_number]

    for other_numb in other_atom_numbers:
        index = str(atom_number) + str(other_numb)
        for axis in ['x', 'y', 'z']:
            s_coord[index + axis] = row[str(atom_number) + axis] - row[str(other_numb) + axis]

    Rel_matrix = []
    cur_vector = []

    for (i, elem) in enumerate(s_coord.values):
        if i % 3 == 0 and i != 0:
            Rel_matrix.append(cur_vector)
            cur_vector = []

        cur_vector.append(elem)
    Rel_matrix.append(cur_vector)

    # print('rel_dists: ', Rel_matrix)

    return np.array(Rel_matrix)

# Короче надо как-то научиться создавать список, в котором каждые N * step шагов будут выкинуты N подряд идущих чисел - это сразу решит проблему обрезания по частицам и шага по состояниям

def generate_useful_indexes(N, step, length):
    '''
    Дает список из индексов для номеров строчек, которые надо использовать при большем цикле считывания
    '''
    sp = []
    for i in range(0, length, step * N):
        for j in range(0, N):
            sp.append(i + j)
    return sp

def get_rows_for_use_particles(old_N, new_N, length):
    '''
    Дает список из индексов для номеров строчек, которые надо использовать при уменьшении числа частиц
    '''
    sp = []
    for i in range(0, length, old_N):
        for j in range(0, new_N):
            sp.append(i + j)
    return sp


# NUMPY VERSION:

def create_csv_from_force(write_folder, read_path, recalculate_forces=False, normalize_forces=False, use_particles=None, step=1, lines_read_coef=None, velocity_regime=None):
    '''
    создает .csv формат из .force
    по-сути делает цсв-хи с которыми я работаю из LAMMPS-овского аутпута

    use_particles - количество частиц, которое использовать, то есть сколько из записанных координат использовать (это нормально реализовать супер геморрой)
    recalculate_forces - пересчитать силы
    normalize_forces - нормализовать силы

    step - шаг на количество позиций при чтении (оно в текущей версии очень долго работает с этим параметром)
    lines_read_coef - N * lines_read_coef строчек с координатами считывается - то есть lines_read_coed - количество конфигураций, которое считывается
    '''
    # через решейп к (lines_read, 3) - удаляем с шагом строчки: x = np.delete(x, np.arange(0, x.size, use_particles))

    if use_particles and not velocity_regime:
        recalculate_forces = True
        print("use_particles is not None - forces will be recalculated anyway")
    with open(read_path, 'r+') as f:
        for i in range(3):
            f.readline()

        N = int(str(f.readline()).strip())

    actual_steps = generate_useful_indexes(N, step, length=int(5e5))

    with open(read_path, 'r+') as read_f:
        all_forces = []
        all_coords = []
        lines_read = 0  # строчки с координатами прочитанные
        for line in (read_f):
            if line[0] == 'C':
                # if lines_read in actual_steps:
                    # if lines_read % (N * step) == 0:    # делаем шаг
                    arr = list(map(lambda x: float(x.strip()), line.split(' ')[1:]))
                    arr_coords = (arr[:3])
                    if not recalculate_forces:
                        arr_forces = (arr[3:])
                        all_forces.extend(arr_forces)
                    all_coords.extend(arr_coords)
                    lines_read += 1

            if lines_read_coef and lines_read >= lines_read_coef * N:
                break
    all_coords = np.reshape(all_coords, (lines_read // N, 3 * N))
    if not recalculate_forces:
        all_forces = np.reshape(all_forces, (lines_read // N, 3 * N))

    if use_particles is not None and use_particles < N:
        # Силы здесь не надо откидывать - если use_particles - их надо пересчитывать
        coords_single_vecs = np.reshape(all_coords, (lines_read, 3))
        length = len(coords_single_vecs)

        new_rows_idxs = get_rows_for_use_particles(old_N=N, new_N=use_particles, length=length)
        coords_single_vecs = coords_single_vecs[new_rows_idxs]

        new_lines_read = len(coords_single_vecs)
        all_coords = np.reshape(coords_single_vecs, (new_lines_read // use_particles, 3 * use_particles))
        N = use_particles
        lines_read = new_lines_read
        CFG.N = use_particles

    # CFG.N = N
    coords_path = write_folder + '/coords' + str(N) + '.csv'
    forces_path = write_folder + '/forces' + str(N) + '.csv'
    if velocity_regime:
        forces_path = write_folder + '/velocities' + str(N) + '.csv'

    fieldnames_forces = []
    fieldnames_coords = []
    for i in range(N):
        fieldnames_coords.extend([str(i) + 'x', str(i) + 'y', str(i) + 'z'])
        if not velocity_regime:
            fieldnames_forces.extend([str(i) + "f_x", str(i) + "f_y", str(i) + "f_z"])
        else:
            fieldnames_forces.extend([str(i) + "v_x", str(i) + "v_y", str(i) + "v_z"])

    if not velocity_regime:
        df_coords = pd.DataFrame(all_coords,
                    index=np.arange(len(all_coords)),
                    columns=fieldnames_coords)
        df_coords.index.name = 't'
        df_coords.to_csv(coords_path)

    if recalculate_forces and not velocity_regime:
        all_forces = []
        for index in tqdm(range(len(df_coords)), desc='Progress for rows: Forces recalculation:'):
            for atom_number in range(N):
                Rel_dists_mat = get_rel_dists(df_coords.loc[index], atom_number, N=N)
                f = np.sum(np.apply_along_axis(_force, -1, Rel_dists_mat), axis=0)
                
                all_forces.append(f)

        all_forces  = np.vstack(all_forces)
        all_forces = np.reshape(all_forces, (lines_read // N, 3 * N))        

    if MODE == "movements" and velocity_regime:
        all_forces = all_forces[:-1]    # потому что перемещения для всех строчек кроме последней определяются

    df_forces = pd.DataFrame(all_forces,
                index=np.arange(len(all_forces)),
                columns=fieldnames_forces)
    df_forces.index.name = 't'
    df_forces.to_csv(forces_path)