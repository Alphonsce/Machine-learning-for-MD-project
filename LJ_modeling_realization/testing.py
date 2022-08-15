import numpy as np
import random
import sys
sys.path.append('./LJ_modeling_realization/includes')
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr
from numpy.linalg import norm
from math import pow, ceil, sqrt
import seaborn as sns
import csv

from includes.constants import *
from includes.calculations import N_grid

line = "123.csv"

print(
    line.split('.')[0]
)