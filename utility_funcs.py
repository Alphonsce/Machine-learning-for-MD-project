import torch
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

MODE = "forces"
# MODE = "movements"
# MODE = "velocities"

naming_of_target_in_csv = {
    "forces": ["f_x", "f_y", "f_z"],
    "movements": ["s_x", "s_y", "s_z"],
    "velocities": ["v_x", "v_y", "v_z"]
}

def create_dataloaders(train_dataset, val_dataset, train_bs=64, val_bs=64):
    '''

    Returns train_loader, val_loader

    '''
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False)

    return train_loader, val_loader

def recieve_loaders(batch_size=64, take_one_projection_for_data=None, path=None, cut_size=None, even_for_train=False):
    '''
    returns: (dataet, train_loader, val_loader), if u pass path -> returns loader from tensor dataset from

    может создать датасет у которого всего одна проекция взята за таргет из уже существующего датасета или по пути на файл:
    take_one_projection_for_data - указываем в этом параметре номер координатной проекции (нумерация от 0)

    even_for_train - брать в качестве трейна четные, в качестве теста нечетные
    '''
    if path:
        N = int(path.split("/")[-1].split('_')[0])     # число атомов
        K = int(path.split("/")[-1].split('_')[-1].split('.')[0])     # можно называть это разрешением...чем число больше, тем больше размеры матрицы для атомов, фактически это число элементов в наборах p и r_cut

        dataset = torch.load("./dataset_objects/" + MODE + '/' + str(N) + '_dataset_K_' + str(K) + '.pt')
        dataset = [(elem[0], elem[1], elem[2], elem[3]) for elem in dataset]

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
        return train_data, val_data, train_dataloader, val_dataloader
    
    X = []
    Y = []

    for _ in range(500):
        # X = (torch.rand(1)).squeeze().unsqueeze(dim=0)
        x = (torch.rand(K))
        y = function(x)
        X.append(x)
        Y.append(y)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42, train_size=0.8)

    X_train = torch.stack(X_train)
    X_val = torch.stack(X_val)
    Y_train = torch.stack(Y_train)
    Y_val = torch.stack(Y_val)

    train_data = TensorDataset(X_train, Y_train)
    val_data = TensorDataset(X_val, Y_val)

    train_dataloader = DataLoader(train_data, batch_size=128)
    val_dataloader = DataLoader(val_data, batch_size=128)

    return train_data, val_data, train_dataloader, val_dataloader

def plot_2d_result(x, y_pred, y_true, figsize=(12, 7)):
    '''
    
    '''
    plt.figure(figsize=figsize)
    
    plt.xlabel('Элемент матрицы X', fontsize=20)
    plt.ylabel(f'Компонента {MODE[:-1]}', fontsize=20)

    plt.scatter(x, y_pred, label='Предсказанная зависимость')
    plt.scatter(x, y_true, label='Истинная зависимость')
    plt.legend(loc='best', fontsize=20)

    plt.show()