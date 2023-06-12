import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn
import multiprocessing
import time
from scipy.special import erf

# See the thesis by Mart van Vliet for more information on the equations that led to this code,
# including how to use and change the variables.

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This file is a toy model to explain the effects of having multiple antenna couples on the internal and external noise, and also on the minimum of the total noise.
# It needs an incoming spectrum, and returns a plot in which the internal and the external noise is plotted against the difference in powers (delta) of the two absorption peaks.
# This way, we find an optimal delta which sits right where the two different noises intersect. 
# For more explanation on how the Ornstein Uhlenbeck process is created around the incoming spectrum, take a look at the file OU_process_CORRECT.py.
# The variables used:
    # data: list, the incoming spectrum at the plant, copied from Arp et. al.
    # w: value in nanometers, the widht of the absorption peaks.
    # s: value, the noise used for the OU process.
    # deltas: array containing the different deltas we want to plot the internal and external noise for. 
    # r: value between 0 and 1, the correlation coefficient for the OU process.
    # n: integer value, the number of antenna couples.
    
with open('C:/Users/martv/Documents/Nat_scriptie/qntenna-master/qntenna-master/spectra/raw_spectra/1m Buiteveld and Kou.txt') as fp:
    data = [list(map(float, line.strip().split(' '))) for line in fp]
    
data = np.transpose(data)
index_ninehundred = data[0].tolist().index(900)
l_a = data[0].tolist()[:index_ninehundred]
mu = data[1].tolist()[:index_ninehundred]

powers = np.zeros((len(data[0].tolist())))
w = 13
s = 0.2
theta = 0.18
deltas = np.linspace(0,0.3,100).tolist()
r = 0.99
b = np.log(r)

# See what happens when changing the amount of antenna couples
n = 10

# Compute the variance of the power from an antenna
def func(w, b, a):
    grens_onder = a - 3*w
    grens_boven = a + 3*w
    Integral = integrate.dblquad(lambda x, y: np.exp(-(x - a)**2/(2*w**2))*np.exp(-(y - a)**2/(2*w**2))*np.exp(-b*np.abs(x - y)), grens_onder, grens_boven, lambda x: grens_onder, lambda x: grens_boven)
    s = 1/(w**2*2*np.pi) * Integral[0]
    return s

Var_PA = s**2/(2*theta)*func(w, b, 400)
print(Var_PA)

# First we compute the three integrals needed to calculate the internal and external noise.
# Then we compute the internal and external noise for different deltas and plot them against chosen deltas.
Int_buiten = []
Int_binnen = []
for d in deltas:
    P_dPb = 1/2*(1 - erf(d/(np.sqrt(2)*Var_PA)))
    E_dPb = Var_PA/(np.sqrt(2*np.pi))*np.exp(-d**2/(2*Var_PA**2))
    E_dPb_squared = -1/(2*np.sqrt(np.pi))*(Var_PA*np.exp(-d**2/(2*Var_PA**2))*((np.sqrt(np.pi)*erf(d/(np.sqrt(2)*Var_PA)) - np.sqrt(np.pi))*Var_PA*np.exp(d**2/(2*Var_PA**2))-np.sqrt(2)*d))
    Int_binnen.append(n*d**2*(1- 2*P_dPb) - n*r*Var_PA**2 + 2*n*(1-r)*d*E_dPb + 2*n*r*E_dPb_squared)
    Int_buiten.append(2*n**2*d**2*P_dPb + 2*n**2*E_dPb_squared - 4*n**2*d*E_dPb )

total = np.zeros(len(Int_buiten)-1)
for i in range(0, len(Int_buiten) - 1):
    total[i] = Int_buiten[i] + Int_binnen[i]

print(deltas[total.tolist().index(min(total))])

plt.figure()
plt.plot(deltas, Int_buiten, label = 'o-t-w')
plt.plot(deltas, Int_binnen, label = 'i-t-w')
plt.plot(0,s, alpha = 0, label = '$\sigma$ = {}'.format(s))
plt.plot(0,r, alpha = 0, label = '$r$ = {}'.format(r))
plt.plot(0,theta, alpha = 0, label = 't = {}'.format(theta))
plt.plot(0,n, alpha = 0, label = 'n = {}'.format(n))
plt.plot(0,w, alpha = 0, label = 'w = {}'.format(w))
plt.xlabel('$\Delta$-bar')
plt.ylabel('Noise')
plt.ylim(0,1)
plt.legend()
