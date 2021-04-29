import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
c = 0
f = lambda x: 0*x #wymuszenie

x_a = 0
x_b = 1
x = 5 #ilosc wezlow NODES

twb_L = 'D'
twb_R = 'D'

wwb_L = 0
wwb_R = 1

WB = [{"ind":1, "typ":'D', "wartosc":1},
      {"ind":x, "typ":'D', "wartosc":2}]

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


def FunkcjeBazowe(x):
    #x stpoień funkcji kształtu
    #zwraca liste funkcji kształtu

    if x==0:
        f = (lambda x: 1+0*x)
        df = (lambda x: 0*x)

    elif x==1:

        f = (lambda x: -1/2*x + 1/2, lambda x: 0.5*x + 0.5)
        df = (lambda x: -1/2 + 0 * x, lambda x: 0.5 + 0 * x )

    #elif n==2:

        #f = (lambda,lambda,lambda)
    else:
        raise Exception("Błąd")

    return f,df
stopien_funkcji_bazowych = 1
phi,dphi = FunkcjeBazowe(stopien_funkcji_bazowych)

print(phi)
print(dphi)

xx = np.linspace(-1,1,101)
plt.plot(xx,phi[0](xx),'r')
plt.plot(xx,phi[1](xx),'g')
plt.plot(xx,dphi[0](xx),'b')
plt.plot(xx,dphi[1](xx),'c')
plt.show()

#Preprocesing

liczbaElementow = np.shape(elementy)[0]
for ee in np.arange(0,liczbaElementow):

    elemRowInd = ee
    elemGlobalInd = elementy[ee,0]

    elemWezel1 = elementy[ee,1]  #indeks wezla poczatkowego elementu ee
    elemWezel2 = elementy[ee,2] # indeks wezla koncowego elementu ee
    indGlobalneWezlow = np.array([elemWezel1, elemWezel2])

    Ml = np.zeros([stopien_funkcji_bazowych + 1, stopien_funkcji_bazowych + 1])

    def Aij(df_i, df_j, c, f_i, f_j):

        fun_podc = lambda x: -df_i(x)*df_j(x) + c * f_i(x)*f_j(x)

        return fun_podc

    p_a = wezly[elemWezel1-1,1]
    p_b = wezly[elemWezel2-1,1]

    J = (p_b-p_a)/2

    m = 0;
    n = 0;

    Ml[m,n] = J * spint.quad(Aij(dphi[m],dphi[n],c,phi[m],phi[n]),-1,1)[0]

    m = 0;
    n = 1;
    Ml[m,n] = J * spint.quad(Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]

    m = 1;
    n = 0;
    Ml[m, n] = J * spint.quad(Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]

    m = 1;
    n = 1;
    Ml[m, n] = J * spint.quad(Aij(dphi[m], dphi[n], c, phi[m], phi[n]), -1, 1)[0]



    A[np.ix_(indGlobalneWezlow-1, indGlobalneWezlow-1)] = \
        A[np.ix_(indGlobalneWezlow - 1, indGlobalneWezlow - 1)] + Ml


    #Uwzględnienie warunków brzegowych

    if WB[0]['typ'] == 'D':
        ind_wezla = WB[0]['ind']
        wart_war_brzeg = WB[0]['wartosc']

        iwp = ind_wezla-1

        WZMACNIACZ = 10**14

        b[iwp] = A[iwp,iwp]*WZMACNIACZ*wart_war_brzeg
        A[iwp,iwp] = A[iwp,iwp]*WZMACNIACZ

    if WB[1]['typ'] == 'D':
        ind_wezla = WB[1]['ind']
        wart_war_brzeg = WB[1]['wartosc']

        iwp = ind_wezla - 1

        WZMACNIACZ = 10**14

        b[iwp] = A[iwp, iwp] * WZMACNIACZ * wart_war_brzeg
        A[iwp, iwp] = A[iwp, iwp] * WZMACNIACZ


    if WB[0]['typ'] == 'N':
        print("Nie zaimplementowano")

    if WB[1]['typ'] == 'N':
        print("Nie zaimplementowano")

    print(A)
    print(b)

    #Rozw liniowych

    u = np.linalg.solve(A,b)

    def RysujRozwiaznie(WEZLY, ELEMENTY, WB, u):

        rysuj(wezly)

        x = NODES[:,1]
        y = u

        plt.plot(x,u,'m*')

    RysujRozwiaznie(wezly, elementy, WB, u)























