import numpy as np
import matplotlib.pyplot as plt

c = 0
f = lambda x: 0*x #wymuszenie

x_a = 4
x_b = 10
x = 7 #ilosc wezlow NODES

twb_L = 'D'
twb_R = 'D'

wwb_L = 0
wwb_R = 1

WB = [{"ind":1, "typ":'D', "wartosc":1},
      {"ind":2, "typ":'D', "wartosc":2}]

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

def Alokacja(x):
    tmp = (x,x)
    tmp1 = (x,1)
    A = np.zeros(tmp)
    b = np.zeros(tmp1)
    return A, b

A,b = Alokacja(x)

print(A)
print(b)


#def Bazowa(x):






