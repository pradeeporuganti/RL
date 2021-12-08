from casadi import *
import csv
import numpy as np

def read_files():
    file = open('u_cl.csv')
    file1 = open('xx.csv')

    csvreader = csv.reader(file)
    csvreader1 = csv.reader(file1)

    u_cl = np.reshape(np.array(list(map(np.float32, next(csvreader)))), (1, 2))
    xx = np.reshape(np.array(list(map(np.float32, next(csvreader1)))), (1, 6))

    while True:
        try:
            u_cl_next = np.reshape(np.array(list(map(np.float32, next(csvreader)))), (1, 2))
        except:
            break
        u_cl = np.concatenate((u_cl, u_cl_next), axis=0)

    while True:
        try:
            xx_next = np.reshape(np.array(list(map(np.float32, next(csvreader1)))), (1, 6))
        except:
            break
        xx = np.concatenate((xx, xx_next), axis=0)

    return u_cl, xx