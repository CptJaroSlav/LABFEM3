import numpy as np
import matplotlib.pyplot as plt

c = 0
f = 0
n = 100

wezly = np.array([[1, 0],
                  [2, 1],
                  [3, 0.5],
                  [4, 0.75]])

elementy = np.array([[1, 1, 3],
                     [2, 4, 2],
                     [3, 3, 4]])

twb_L = 'D'
twb_R = 'D'

wwb_L = 0
wwb_R = 1

def genTABLICEGEO(x_a, x_b, x):

    temp = (x_b-x_a)/(x-1)
    array = np.array([1,0])

    for i in range(1, x, 1):
        array = np.block([
            [array],
            [i+1, i * temp + x_a],
        ])
    return array


print(genTABLICEGEO(1,4,4))


