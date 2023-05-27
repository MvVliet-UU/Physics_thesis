# Physics_thesis
Code used for the thesis of Mart van Vliet, June 2023

This repository consists of three files, containing all the computations needed to follow the steps that were done in the thesis. \
#1. OU_Process_CORRECT.py \
This file can compute a line on which every point behaves as an Ornstein Uhlenbeck process, and all the Ornstein Uhlenbeck processes are correlated to create a wobbling spectrum, that fluctuates around a mean spectrum. Furthermore, it draws powers from two antennas and shows an animation of how the powers change over time, with respect to the wanted power output at the reaction centre of the plant. 
#2. WerkendPlot_uitrekenen_optimal_lambdas.py 
This python script is the main file, it can compute the noise given the incoming spectrum and the place of the two antennas on the spectrum. It returns a heatmap in which different areas come up with different noise values.
#3. binnen_vs_buiten.py 
This last file is a toy model to give an intuitive idea of how multiple antenna couples can change the internal and external noise, and how these two noises differ relatively to each other for different deltas in power between the two antennas.
