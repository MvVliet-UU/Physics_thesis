import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn
import multiprocessing
import time
from scipy.special import erf

# See the thesis by Mart van Vliet for more information on the equations that led to this code,
# including how to use and change the variables.

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Check the file OU_process_CORRECT.py, which explains how to choose the variables so a realistic Ornstein Uhlenbeck process (OU process) is created.
# This file takes in the parameters for the OU process, combined with the spectral data from Arp, and converts this into a heatmap.
# This heatmap sets center wavelength lambda_0 against the peak seperation \delta lambda and shows the total noise.
# First, we adjust the spectrum so that it is easier to work with.
# We check the variables:
    # data: list containing the spectral data used by Arp et. al

# Open the file with the spectral data and substract mu, the mean spectrum around which we have created the OU process.
# with open('C:/Users/martv/Documents/Nat_scriptie/qntenna-master/qntenna-master/spectra/NREL-visible.txt') as fp:
# with open('C:/Users/martv/Documents/Nat_scriptie/qntenna-master/qntenna-master/spectra/raw_spectra/10cm Buiteveld and Kou.txt') as fp:
#     data = [list(map(float, line.strip().split(' '))) for line in fp]
  
# with open('C:/Users/martv/Documents/Nat_scriptie/qntenna-master/qntenna-master/spectra/raw_spectra/5m Buiteveld and Kou.txt') as fp:
with open('C:/Users/martv/Documents/Nat_scriptie/qntenna-master/qntenna-master/spectra/raw_spectra/NREL_ASTM_Direct_Circum.txt') as fp:
    data = [list(map(float, line.strip().split('\t'))) for line in fp]

data = np.transpose(data)
index_ninehundred = data[0].tolist().index(701)
l_a = data[0].tolist()[:index_ninehundred]
mu = data[1].tolist()[:index_ninehundred]

# This step is a bit tricky and is specificly used for the NREL-visible spectrum.
# It removes all the data points that sit between two points with wavelength 1 nm apart.
# The goal is to create a spectrum with where every wavelength is 1 nm away from neighbours.
# This fails if the data points differ by a wavelength greater than 1 nm.
for x in np.arange(0,100):
    if l_a[x].is_integer() == True:
        continue
    else:
        l_a.pop(x)
        mu.pop(x)

# Now we have to extend the outer points of the spectrum with 50 nm, otherwise, the heatmap cannot be computed for absorption peaks on the edges of the spectrum.
l_b_gaussian = np.arange(l_a[0] - 50, l_a[0], 1).tolist() + l_a + np.arange(l_a[-1]+1, l_a[-1] + 51, 1).tolist()
l_a_gaussian = l_b_gaussian
extra_mu_1 = np.ones(50)*mu[0]
extra_mu_2 = np.ones(50)*mu[-1]
mu_gaussian = extra_mu_1.tolist() + mu + extra_mu_2.tolist()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Now we set the mean powers for every possible absorption peak in an array
# This makes it easier to extract the powers for two different absorption peaks.
# We check the variables:
    # s: value, the noise of the OU process. Best and most realistic results with s between 0.05 and 1
    # theta: value, the mean reversion speed of the OU process. Most realistic between 0 and 1.
    # n: integer value, the amount of antenna couples to compute the heatmap for.
    # w: value in nanometers, the width or standard deviation of the absorption peaks.
    # rho: value between 0 and 1, the correlation coefficient between two points on the spectrum. Best and most realistic results between 0.9 and 1.

s = 0.18
theta = 0.2
n = 10
w = 13
rho = 0.99
b_rho = np.log(rho)

# We create the array that will contain the powers for every possible centre l_a of the absorption peak.
j = 0
powers = np.zeros(len(l_a))
gaussians = np.zeros((len(l_a), len(l_b_gaussian)))
for l in l_a:
    gaussian = np.zeros(len(l_a_gaussian))
    i = 0
    for x in l_b_gaussian:
        gaussian[i] = (1/(w*np.sqrt(2*np.pi))*np.exp(-(x - l)**2/(2*w**2)))
        i+=1
    # gaussians[l_a.index(l)] = gaussian
    powers[j] = np.sum(gaussian*mu_gaussian)
    j+=1


# # A function to compute the integral given in appendix A of the stated thesis, with it we can compute the variance of the power of an absorption peak (is the same for every peak centre l_a).
# def func(w, b_rho, a):
#     grens_onder = a - 3*w
#     grens_boven = a + 3*w
#     Integral = integrate.dblquad(lambda x, y: np.exp(-(x - a)**2/(2*w**2))*np.exp(-(y - a)**2/(2*w**2))*np.exp(-b_rho*np.abs(x - y)), grens_onder, grens_boven, lambda x: grens_onder, lambda x: grens_boven)
#     s = 1/(w**2*2*np.pi) * Integral[0]
#     return s

Var_PA = s**2/(theta)*np.exp(b_rho**2*w**2)*1/2*(1-erf(b_rho*w/np.sqrt(2)))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Now we set up the variables to create the heatmap, after which the heatmap is plotted. 
# It takes into account the external fluctuations, Int_buiten, and the internal fluctuations, Int_binnen and adds these up.
# To calculate the fluctuations, three integrals, P_dPb, E_dPb, E_dPb_squared, were needed, the names are selfexplanatory when combined with the formulas in the thesis
# The integrals were written to error functions to significantly reduce computation time.
# The variables:
    # l_diff: array with values in nanometer. It contains the peak seperation, the y-axis of the heatmap, 
    # l_null: array, containing the adjusted spectrum.
    
optimal_power = 0.1161616
u = 0
optimal_powers = []
for l in l_a:
    optimal_diff = 10000
    for v in np.arange(0, 71, 1):
        if u - v < 0:
            powerdiff = np.abs(powers[0] - powers[u + v])
            if np.abs(powerdiff - optimal_power) < optimal_diff:
                optimal_diff = np.abs(powerdiff - optimal_power)
                ind_v = v
            else:
                continue
        if u - v > 0 and u + v < len(powers) - 1:
            powerdiff = np.abs(powers[u - v] - powers[u + v])
            if np.abs(powerdiff - optimal_power) < optimal_diff:
                optimal_diff = np.abs(powerdiff - optimal_power)
                ind_v = v
            else:
                continue
        if u + v > len(powers) - 1:
            powerdiff = np.abs(powers[u - v] - powers[len(powers)-1])
            if np.abs(powerdiff - optimal_power) < optimal_diff:
                optimal_diff = np.abs(powerdiff - optimal_power)
                ind_v = v
            else:
                continue
        
    u+=1
    optimal_powers.append(ind_v)

optimal_powers = [2*z for z in optimal_powers]

p = 0
worst_powers = []
for l in l_a:
    optimal_power = 1000
    for q in np.arange(1, 71, 1):
        if p - q < 0:
            powerdiff = np.abs(powers[0] - powers[p + q])
            if powerdiff < optimal_power:
                optimal_power = powerdiff
                ind_q = q
            else:
                continue
        if p - q > 0 and p + q < len(powers) - 1:
            powerdiff = np.abs(powers[p - q] - powers[p + q])
            if powerdiff < optimal_power:
                optimal_power = powerdiff
                ind_q = q
            else:
                continue
        if p + q > len(powers) - 1:
            powerdiff = np.abs(powers[p - q] - powers[len(powers)-1])
            if powerdiff < optimal_power:
                optimal_power = powerdiff
                ind_q = q
            else:
                continue
        
    p+=1
    worst_powers.append(ind_q)

worst_powers = [2*z for z in worst_powers]

    
    
l_diff = np.arange(0, 141, 1).tolist()
l_null = l_a
j = np.arange(0, len(l_null), 1)
i = np.arange(0, len(l_diff), 1)

# The next function is need for computing the peak centres a, b of absorption peaks for antenna A and B resp.
# If the peak centre is not an integer wavelength, find the nearest integer value in the possible set of wavelengths.
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Now we create the, yet empty, noise matrix containing the values for the heatmap. 
noise = np.zeros(((len(l_diff)), len(l_null)))

# When checking the internal and external noise seperately, the following empty matrices can be used.
# Int_binnen = np.zeros(((len(l_diff)), len(l_null)))
# Int_buiten = np.zeros(((len(l_diff)), len(l_null)))

# The next for loop firstly computes the difference d in power, the correlation coefficient r for every two absorption peak centres a and b.
# Next, it computes P_dPb, E_dPb and E_dPb_squared from which it calculates the expected internal and external fluctuations.
correlations = []
for d_l in l_diff:
    Covariance = s**2/(2*theta)*(np.exp(b_rho*d_l + b_rho**2*w**2)*1/2*(1 - erf((b_rho*w-d_l/(2*w))/np.sqrt(2))) + np.exp(-b_rho*d_l + b_rho**2*w**2)*1/2*(1-erf((b_rho*w+d_l/(2*w))/np.sqrt(2))))
    correlations.append(Covariance/Var_PA)


for x in l_null:
    for y in l_diff:
        a = x-y/2
        b = x+y/2
        
        
        try:
            ind_a = l_a.index(a)
        except ValueError:
            ind_a = l_a.index(find_nearest(l_a, a))

        try:
            ind_b = l_a.index(b)
        except ValueError:
            ind_b = l_a.index(find_nearest(l_a, b))

        d = np.abs(powers[ind_a] - powers[ind_b])
        r = correlations[y]
        
        P_dPb = 1/2*(1 - erf(d/(np.sqrt(2)*Var_PA)))
        E_dPb = Var_PA/(np.sqrt(2*np.pi))*np.exp(-d**2/(2*Var_PA**2))
        E_dPb_squared = -1/(2*np.sqrt(np.pi))*(Var_PA*np.exp(-d**2/(2*Var_PA**2))*((np.sqrt(np.pi)*erf(d/(np.sqrt(2)*Var_PA)) - np.sqrt(np.pi))*Var_PA*np.exp(d**2/(2*Var_PA**2))-np.sqrt(2)*d))

        Int_binnen = n*d**2*(1- 2*P_dPb) - n*r*Var_PA**2 + 2*n*(1-r)*d*E_dPb + 2*n*r*E_dPb_squared
        Int_buiten = 2*n**2*d**2*P_dPb + 2*n**2*E_dPb_squared - 4*n**2*d*E_dPb 

        # The following matrices can be filled when computing the internal or external noise seperately.
        # Int_binnen[l_diff.index(y)][l_null.index(x)] = d**2 - r*s**2 - 2*((d**2*P_dPb) + (d*(r-1)*E_dPb) - (r*E_dPb_squared))
        # Int_buiten[l_diff.index(y)][l_null.index(x)] = 2*((d**2*P_dPb) - (2*d*E_dPb) + (E_dPb_squared))
        
        # The total noise for the two absorption peaks is the sum of expected internal and external fluctuations.
        noise[l_diff.index(y)][l_null.index(x)] = Int_buiten + Int_binnen

mu = np.array(mu)

# The following three lines can compute the two absorption peaks in the red and blue part that have minimal noise, which is not really useful since a lot of points in the heatmap have noise with almost the same value as the minimum.  
best = noise.min()
ind_min = np.where(noise == best)
ind_min = tuple([i.item() for i in ind_min])

second_min_noise = noise[:, 241:]
best_2 = second_min_noise.min()
ind_min_2 = np.where(noise == best_2)
ind_min_2 = tuple([i.item() for i in ind_min_2])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Now we plot the noise heatmap, called the optimization landscape
# Furthermore, in green we plot the mean incoming spectrum, which makes the heatmap easier to interpret. 
# In the legend are the used values for the OU process and the absorption peaks stated.
plt.figure()
plt.title('Optimization landscape of the total noise from $n$ antenna pigments')
# ax = seaborn.heatmap(Int_buiten)
# ax = seaborn.heatmap(Int_binnen)
ax = seaborn.heatmap(noise, norm = colors.LogNorm(vmin = 0.1, vmax = 1.2), cbar_kws = {'ticks': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]} )
# ax = seaborn.heatmap(noise, norm = colors.LogNorm(vmin = 0.55, vmax = 4), cbar_kws = {'ticks': [0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]} )

ax.xaxis.set_ticks(j[::50])
ax.xaxis.set_ticklabels(l_null[::50])
ax.set_xlabel('$\lambda_0$ (nm)')
ax.yaxis.set_ticks(i[::20])
ax.yaxis.set_ticklabels(l_diff[::20])
ax.set_ylabel('$\lambda_A - \lambda_B$ (nm)')
ax.invert_yaxis()
ax.collections[0].colorbar.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
# ax.collections[0].colorbar.set_ticklabels([0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4])
plt.plot(np.arange(0, len(l_a), 1), 85*mu - np.ones(len(l_a))*(min(85*mu)), c = 'black', linewidth = 1.5)
plt.plot(0,s, alpha = 0, label = '$\sigma$ = {}'.format(s))
plt.plot(0,rho, alpha = 0, label = '$r$ = {}'.format(rho))
plt.plot(0,theta, alpha = 0, label = '$theta$ = {}'.format(theta))
plt.plot(0,n, alpha = 0, label = 'n = {}'.format(n))
plt.plot(0,w, alpha = 0, label = 'w = {}'.format(w))
# plt.plot(np.arange(0, len(l_a), 1), optimal_powers, '.' , c = 'blue', linewidth = 2)
# plt.plot(np.arange(0, len(l_a), 1), worst_powers, '.' , c = 'brown', linewidth = 2)
# plt.plot(np.arange(0, len(l_a), 1), 85*powers, c = 'magenta', linewidth = 2)
plt.legend()


# An extra feature is to plot the contour lines in the heatmap, this is optional.
contour = plt.contour(noise, levels = [0, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2])
plt.clabel(contour, levels = [0, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2])
# contour = plt.contour(noise, levels = [0, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3])
# plt.clabel(contour, levels = [0, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3])
plt.plot(ind_min[1], ind_min[0], 'o', c = 'red')
plt.plot(ind_min_2[1], ind_min_2[0], 'o', c = 'red')

# The first number is the optimal wavelength the different.
# Check in the l_a list the wavelength that corresponds with the second number of the printed indices. 
print(ind_min, ind_min_2)
