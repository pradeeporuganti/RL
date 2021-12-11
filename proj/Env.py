from casadi import *
import numpy as np


# Discrete vehicle dynamics
F = external('F', './vehd.so')

print(F([0, 0, 0, 4, 0], np.array([0])))

