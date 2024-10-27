# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:02:32 2024

@author: Luis Lucas García
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
"""
Vamos a programar una simulación de la ecuación de Schrödinger usando el método
de Crank-Nicolson. Haremos una programación de forma inteligente, de modo que
podamos simular varios potenciales usando funciones de simulación.

Exigiremos que la función de onda se anule en los extremos de nuestro intervalo,
esto puede interpretarse como una exigencia de la normalización, para que la integral
converja. Para que esto no afecte mucho el tratamiento en otros casos podemos tomar
valores más grandes del intervalo, es como tener dentro de un pozo infinito todos los
potenciales.
"""
plt.close("all")

def buildMatrix(d, o, u): #Builds a tridiagonal matrix from its diagonals

    n = len(d)
    M = np.zeros((n, n), dtype=complex)
    M[0][0] = d[0]
    
    for i in range(1, n):
        
        M[i][i] = d[i]
        M[i-1][i] = o[i-1]
        M[i][i-1] = u[i-1]
    
    return M

#Vamos a definir la función que resuelve la ec. de Schrödinger

def solveSchrodinger(U, psix0, dx, dt, tmax=0.6, xmin=0, xmax=1): #U y psi0 funciones
    
    x = np.arange(xmin, xmax+dx, dx)
    t = np.arange(0, tmax+dt, dt)
    nx = len(x)
    nt = len(t)
    u = np.array([U(i) for i in x])
    psi0 = np.array([psix0(i) for i in x])
    psi0[0] = 0
    psi0[nx-1] = 0
    psi = [psi0]
    
    #Tenemos dos matrices a usar para CN, las calculamos
    
    r = 1j*dt/(4*dx**2)
    b = 1j*dt*u/2
    
    dA = np.ones(nx) + b +2*r*np.ones(nx)
    dA[0] = 1
    dA[nx-1] = 1
    uA = -r*np.ones(nx-1)
    uA[nx-2] = 0
    oA = -r*np.ones(nx-1)
    oA[0] = 0
    
    A = buildMatrix(dA, oA, uA)
    
    dB = np.ones(nx) -b - 2*r*np.ones(nx)
    dB[0] = 1
    dB[nx-1] = 1
    uB = r*np.ones(nx-1)
    uB[nx-2] = 0
    oB = r*np.ones(nx-1)
    oB[0] = 0
    
    B = buildMatrix(dB, oB, uB)
    
    #Ahora construimos el array de los psis
    
    C = np.dot(np.linalg.inv(A), B)
    
    for i in range(1, nt):
        
        psi.append(np.dot(C, psi[i-1]))
        
    return psi, t, x

def animatePsi(psi, t, x, fig, ax, U, a=1): #Genera la animación de psi
    
    v = [U(i) for i in x]
    ymax = max(np.abs(psi[0]))
    def update(i):
        
        ax.cla()
        ax.set_ylim(-ymax-0.1, ymax+0.1)
        ax.grid()
        ax.set_xlabel("x")
        ax.set_ylabel("$\\psi (x, t)$")
        ax.plot(x, np.real(psi[i]), "--", label="$Re(\\psi)$")
        ax.plot(x, np.imag(psi[i]), "--", label="$Im(\\psi)$")
        ax.plot(x, (np.real(psi[i])**2 + np.imag(psi[i])**2), label="$|\\psi|^2$")
        ax.plot(x, v, label="U")
        ax.legend(loc="upper right")
        
    f = range(0, len(t), a)
    anim = FuncAnimation(fig, update, frames=f)
    
    return anim

def plotPsi(psi, U, x, name):
    
    ymax = max(np.real(psi[0])**2 + np.imag(psi[0])**2)
    v = [U(i) for i in x]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Tiempo inicial")
    plt.ylim(-ymax-0.1, ymax+0.1)
    plt.xlabel("x")
    plt.ylabel("$\\psi$")
    plt.grid()
    plt.plot(x, np.real(psi[0]), "--", label="$Re \\left( \\psi (x, t) \\right)$")
    plt.plot(x, np.imag(psi[0]), "--", label="$Im \\left( \\psi(x, t) \\right)$")
    plt.plot(x, np.abs(psi[0]), label="$|\\psi(x, t=0)|^2$")
    plt.plot(x, v, label="U(x)")
    plt.subplot(1, 2, 2)
    plt.title("Tiempo final")
    plt.xlabel("x")
    plt.ylabel("$\\psi$")
    plt.grid()
    plt.ylim(-ymax-0.1, ymax+0.1)
    plt.plot(x, np.real(psi[len(psi)-1]), "--", label="$Re \\left( \\psi (x, t) \\right)$")
    plt.plot(x, np.imag(psi[len(psi)-1]), "--", label="$Im \\left( \\psi(x, t) \\right)$")
    plt.plot(x, np.abs(psi[len(psi)-1]), label="$|\\psi(x, t)|^2$")
    plt.plot(x, v, label="U(x)")
    plt.legend(loc="upper right")
    plt.savefig(name, dpi=200)
    
def calcEner(psi, t, dx):
    
    nt = len(t)
    ener = np.zeros(nt)
    
    for j in range(nt):
    
        for i in range(len(psi[j])):
            
            ener[j] += dx*np.abs(psi[j][i])**2
            
    return ener
        
dx = 0.01
dt = 0.001

#Pozo infinito

pozo = lambda x: 0
psiPozo0 = lambda x: np.exp((-(x-0.5)**2)/(10e-2**2))
psiPozo, tPozo, x = solveSchrodinger(pozo, psiPozo0, dx, dt)
figPozo = plt.figure()
axPozo = plt.axes()
animPozo = animatePsi(psiPozo, tPozo, x, figPozo, axPozo, pozo)
#animPozo.save("Animaciones/Animación 1 - Caso de prueba.gif", dpi=200)
plotPsi(psiPozo, pozo, x, "Imagenes/Imagen 1 - Caso control.png")
enerPozo =  calcEner(psiPozo, tPozo, dx)

#Oscilador armónico

oscilador = lambda x: 0.5*x**2
psiOsci0 = lambda x: x*np.exp(-0.5*x**2)
psiOsci, tOsci, x = solveSchrodinger(oscilador, psiOsci0, dx, dt, xmin=-4, xmax=4, tmax=1.6)
figOsci = plt.figure()
axOsci = plt.axes()
animOsci = animatePsi(psiOsci, tOsci, x, figOsci, axOsci, oscilador, a=10)
#animOsci.save("Animaciones/Animación 2 - Oscilador primer estado.gif", dpi=200)
plotPsi(psiOsci, oscilador, x, "Imagenes/Imagen 2 - Oscilador armónico.png")
enerOsci =  calcEner(psiOsci, tOsci, dx)

#Paquete de ondas con barrera

bar = lambda x: 40 if -1 < x < 1 else 0
psiBar0 = lambda x: np.exp(-(x+2)**2 + 10j*x)
psiBar, tBar, x = solveSchrodinger(bar, psiBar0, dx, dt, xmin=-8, xmax=8, tmax=0.9)
figBar = plt.figure()
axBar = plt.axes()
animBar = animatePsi(psiBar, tBar, x, figBar, axBar, bar, a=5)
#animBar.save("Animaciones/Animación 3 - Tunelación.gif", dpi=200)
plotPsi(psiBar, bar, x, "Imagenes/Imagen 3 - Tunelacion.png")
enerBar =  calcEner(psiBar, tBar, dx)

#Potencial periódico, peine de Dirac.

def perio(x, l=0.02, a=1, xmin=-10, xmax=10):
    
    x0 = xmin
    lista = [x0]
    
    while x0 < xmax:
        
        x0 = x0 + a
        lista.append(x0)
    
    for i in lista:
        
        if i-l/2 < x < i+l/2:
            
            return 40
        
    return 0

psiPerio0 = lambda x: np.sin(0.25*np.pi*x)
psiPerio, tPerio, x = solveSchrodinger(perio, psiPerio0, dx, dt, xmin=-8, xmax=8)
figPerio = plt.figure()
axPerio = plt.axes()
animPerio = animatePsi(psiPerio, tPerio, x, figPerio, axPerio, perio)
#animPerio.save("Animaciones/Animación 4 - Potencial periódico.gif", dpi=200)
plotPsi(psiPerio, perio, x, "Imagenes/Imagen 4 - Potencial periodico.png")
enerPerio =  calcEner(psiPerio, tPerio, dx)

#Potencial con término centrifugo

eff = lambda x: 6*(1/(x-1)**2 - 2/(x-1) + 1)
psiEff0 = lambda x: np.exp(-(x-3)**2 + 1j*x)
psiEff, tEff, x = solveSchrodinger(eff, psiEff0, dx, dt, xmin=-2.001, xmax=10, tmax=3)
figEff = plt.figure()
axEff = plt.axes()
animEff = animatePsi(psiEff, tEff, x, figEff, axEff, eff, a=30)
#animEff.save("Animaciones/Animación 5 - Paquete de ondas y potencial centrífugo.gif", dpi=200)
plotPsi(psiEff, eff, x, "Imagenes/Imagen 5 - Potencial efectivo.png")
enerEff = calcEner(psiEff, tEff, dx)

plt.figure()
plt.plot(tPozo, enerPozo,  label="Pozo")
plt.plot(tOsci, enerOsci, label="Oscilador")
plt.plot(tPerio, enerPerio, label="Periódico")
plt.plot(tEff, enerEff, label="Efectivo")
plt.plot(tBar, enerBar, label="Barrera")
plt.title("$\\int |\\psi(x, t)|^2 dx$")
plt.xlabel("t")
plt.ylabel("$\\psi$")
plt.grid()
plt.legend()
plt.savefig("Imagenes/Imagen 6 - Modulo cuadrado.png", dpi=200)

plt.show()