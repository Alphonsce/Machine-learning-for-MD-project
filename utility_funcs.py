import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# MODE = "forces"
MODE = "movements"
# MODE = "velocities"

path = './dataset_objects/' + MODE + '/2_dataset_K_3.pt'        # ЗДЕСЬ БЫЛО МОДЕ ВМЕСТО movements
path_vel = './dataset_objects/' + "d_velocities" + '/2_dataset_K_3.pt'        # ЗДЕСЬ БЫЛО МОДЕ ВМЕСТО movements

class CFG:
    '''

    All hyperparameters are here

    '''

    N = int(path.split("/")[-1].split('_')[0])     # число атомов
    K = int(path.split("/")[-1].split('_')[-1].split('.')[0])     # можно называть это разрешением...чем число больше, тем больше размеры матрицы для атомов, фактически это число элементов в наборах p и r_cut

    L = 2 * N ** (1 / 3) # размер одной клетки при моделировании

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