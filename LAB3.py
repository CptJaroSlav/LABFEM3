import numpy as np
import matplotlib.pyplot as plt

c = 0
f = 0

x_a = 0
x_b = 1
x = 8 #ilosc wezlow

twb_L = 'D'
twb_R = 'D'

wwb_L = 0
wwb_R = 1



# wezly = np.array([[1, 0],
#                   [2, 1],
#                   [3, 0.5],
#                   [4, 0.75]])
#
# elementy = np.array([[1, 1, 3],
#                      [2, 4, 2],
#                      [3, 3, 4]])
#
# twb_L = 'D'
# twb_R = 'D'
#
# wwb_L = 0
# wwb_R = 1


def genTABLICEGEO(x_a, x_b, x):

    temp = (x_b-x_a)/(x-1)
    array = np.array([1,x_a])
    array2 = np.array([1,1,2])

    for i in range(1, x, 1):
        array = np.block([
            [array],
            [i+1, i * temp + x_a],
        ])

    for i in range(2, x, 1):
        array2 = np.block([
            [array2],
            [i, i, i + 1]
        ])
    return array, array2


wezly,elementy = genTABLICEGEO(x_a,x_b,x)

print(wezly)
print(elementy)

def rysuj(wezly):
    y = np.zeros(wezly.shape[0])

    plt.plot(wezly[:,1],y,marker='o')

    for i in range(0, np.size(np.zeros(x)), 1):  #wezly
        plt.text(x = wezly[i, 1], y = y[i] - 0.007, s = (wezly[i, 0]), fontsize=7, color = 'green')

    for i in range(0, np.size(y) - 1, 1):  # podpis elementow
        plt.text(x = (wezly[i, 1] + wezly[i + 1, 1]) / 2, y = y[i] + 0.003, s = int(i + 1), fontsize = 7, color = 'blue')


    plt.grid(True)
    plt.show()

rysuj(wezly)
