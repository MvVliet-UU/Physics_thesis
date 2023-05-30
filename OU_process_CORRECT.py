import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 
import csv
from celluloid import Camera

# See the thesis by Mart van Vliet for more information on the equations that led to this code,
# including how to use and change the variables.

# --------------------------------------------------------------------------------------------------------------------
# First, we start by creating an Ornstein Uhlenbeck process (OU process), which we create from an incoming spectrum.
# We explain the variables:
    # data: list, the imported list with data on the incoming spectra at the plant, copied from Arp. et al.
    # mu: array, the mean spectrum at the plant, around which we create the OU process.
    # theta: a value for the mean reversion speed, the higher, the more quickly the process mean reverts. Here a speed between 0 and 1 is the most realistic. 
    # sigma: the noise of the OU process, the bigger sigma, the more fluctuations the spectrum shows, realisticly big changes in the incoming sunlight at the plant.
    # rho: value between 0 and 1, the correlation coefficient between two points on the spectrum.
    # T: total time for the OU process.
    # N: number of timesteps for the OU process.
    

# Opening the map with the spectral data taken from Arp et. al
with open('C:/Users/martv/Documents/Nat_scriptie/qntenna-master/qntenna-master/spectra/NREL-visible.txt') as fp:
    data = [list(map(float, line.strip().split(' '))) for line in fp]

mu_data = np.transpose(data)

# We substract the mean spectrum from the data
mu = mu_data[1]

# Parameters for the Ornstein-Uhlenbeck process, decreasing theta and increasing N should give a OU process that moves more smoothly. 
theta = 0.2
sigma = 0.05
rho = 0.998
b = -np.log(rho)


T = 10.0
N = 100
dt = T / N

# Now we create the matrix that has the spectrum at different timesteps in the rows.
# The rows are the different spectra at different times, and every column represents one wavelength.
points = len(data)
matrix = np.zeros((points, N))

# We create the OU process for one point on the line,
# then let the next point take a noise correlated to the noise of the first point, with correlation coefficient rho.
for j in range(0,points):
    y = np.zeros(N)
    
    # Starting point for every point on the line
    y[0] = mu[j]
    if j ==0:
        dWy = np.random.normal(scale = sigma, size = N)
    else:
        dWy = dWy*(rho) + np.sqrt( 1-rho**2)*np.random.normal(scale = sigma, size = N)
    for t in range(1,N):
        y[t] = y[t-1] - theta*(y[t-1] - mu[j]) + dWy[t]

        
    matrix[j] = y

# Now the matrix has rows containing the intensity of one wavelength for different timesteps,
# while the columns go over all wavelengths, hence we should transpose to get the desired matrix.
matrix = np.transpose(matrix)

# ------------------------------------------------------------------------------------------------------------------
# The next part creates the absorption peaks for two given wavelengths and computes the powers on every timestep.
# The variables:
    # w: value in nanometers, the width or standard deviation of the gaussian absorption peaks.
    # lambda_A, lambda_B: values in nanometers, the position of the centre of the absorption peak for antenna A resp. B.

w = 13

lambda_A = 430
lambda_B = 370

# All the different wavelengths over which the spectrum was measured.
lambdas = mu_data[0]

A_gaussian = np.zeros(points)
B_gaussian = np.zeros(points)
i = 0

# Create the two absorption peaks
for x in lambdas:
    A_gaussian[i] = (1/(w*np.sqrt(2*np.pi))*np.exp(-(x - lambda_A)**2/(2*w**2)))
    B_gaussian[i] = (1/(w*np.sqrt(2*np.pi))*np.exp(-(x - lambda_B)**2/(2*w**2)))
    i += 1

Power_A = np.zeros(points)
Power_B = np.zeros(points)
Tot_Power_A = []
Tot_Power_B = []

# To compute the powers, take the integral stated in the thesis.
# Since we have discrete points in the code, the integral is approximated by a discrete sum over all wavelengths.
for t in range(0, N):
    for l in range(0,points):
        Power_A[l] = A_gaussian[l] * matrix[t][l]
        Power_B[l] = B_gaussian[l] * matrix[t][l]
    Total_A = np.sum(Power_A) 
    Total_B = np.sum(Power_B)
    
    # Assumed in the thesis that the power from A (P_A) is never smaller than the power from B (P_B), so we should switch them if this happens. 
    if Total_A < Total_B:
        z = Total_A
        Total_A = Total_B
        Total_B = z
    Tot_Power_A.append(Total_A)
    Tot_Power_B.append(Total_B)

lambda_nm = mu_data[0]
t = np.linspace(0,T,N)

# Compute the wanted photon rate at the plant.
Omega = (Tot_Power_A[0] + Tot_Power_B[0])/2

# ----------------------------------------------------------------------------------------------------------------------------------------    
# The first plot is an animation of the OU-process created, to check if the chosen variables describe reality accurate.
# The second plot is a figure putting extra attention on the absorption peaks with respect to the mean incoming spectrum.
# The third plot shows the powers changing over time, also plotted is a horizontal line which represents Omega.
# The fourth plot is the same as the third plot, but animated to create the effect of time passing by.

# 1 ----------------------------------------------------------------------------------------------------------
fig = plt.figure()
camera = Camera(fig)
for i in range(N):
    plt.plot(lambda_nm, matrix[i], c = 'blue', label = "$I(\lambda, t)$")
    plt.ylim(0, 2.5)
    camera.snap()
animation = camera.animate(interval = 400)
animation.save('C:/Users/martv/Documents/Nat_scriptie/OU_process/OU_process.gif')


# 2 ----------------------------------------------------------------------------------------------------------
# mu = mu.tolist()
# plt.figure()
# plt.plot(lambda_nm, mu-np.ones(len(mu))*mu[mu.index(min(mu))], c = 'black', label = "$I(\lambda, t)$")
# plt.plot(lambda_nm, 10* A_gaussian, c = 'red', label = "$P_A$")
# plt.plot(lambda_nm, 10* B_gaussian, c = 'orange', label = "$P_B$")
# plt.legend()


# 3 ----------------------------------------------------------------------------------------------------------
plt.figure()
plt.plot(t, Tot_Power_A, c = 'red', label = "$P_A$")
plt.plot(t, Tot_Power_B, c = 'orange', label = "$P_B$")
plt.plot(t, Omega*np.ones(len(t)),  '--', c = 'blue', label = "$\Omega$")
plt.xlabel('time')
plt.legend()


# 4 ----------------------------------------------------------------------------------------------------------
# fig = plt.figure()
# camera = Camera(fig)
# for i in range(N):
#     plt.plot(t[:i], Tot_Power_A[:i], c = 'red', label = "$P_A$")
#     plt.plot(t[:i], Tot_Power_B[:i], c = 'orange', label = "$P_B$")
#     plt.plot(t, Omega*np.ones(len(t)),  '--', c = 'blue', label = "$\Omega$")
#     camera.snap()
# animation = camera.animate(interval = 400)
# animation.save('C:/Users/martv/Documents/Nat_scriptie/OU_process/OU_process_Powers.gif')
