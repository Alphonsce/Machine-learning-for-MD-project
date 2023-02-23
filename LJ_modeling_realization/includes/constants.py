# ---------------------------------- #
# Argon is used as a reference gas, his parameters in SI: m = 6.7E-26 kg; epsilon = 1.6E-21 J; sigma = 3.4E-10 m

SIGMA = 1
EPSILON = 1
M = 1

TIME_STEPS = 10000

N = 4

L = 2 * N ** (1 / 3)
# L = 0.25

# the distance of cut for the LJ potential
r_cut = 10
#rho = N / L ** 3
dt = 0.0001     # 0.001 - small, 0.0005 - OK, 0.0001 - too long to wait, but perfect
# T is in epsilon / k_b units;

# ---------------------------------- #

boundary_conditions = True
boundary_conditions_teleportation = True

# TODO: ДОБАВИТЬ УЧЕТ ГРАНИЧНЫХ УСЛОВИЙ К МАТРИЦЕ X = V @ A