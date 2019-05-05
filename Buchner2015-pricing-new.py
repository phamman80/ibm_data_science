# -*- coding: utf-8 -*-


#A. Buchner (2015) Equilibrium option pricing: A Monte Carlo approach

import os
import math
import numpy as np
np.set_printoptions(precision=3)
import itertools as it
import pandas as pd
from time import time
# Inverse of the CDF of the standard normal distribution 
from scipy.stats import norm
#ENTER PATH:
path =r'D:\Google Drive\M2 P1\Reference\Theorie Option_MYannick\Articles\CODE'


#Heston(1993) Diffusion Parameters
v0 = 0.1
kappa_v = 0.5
# sigma_v = 0.05
rho_vs = -0.5
theta_v = 0.1
S0 = 100
mu_s = 0.1241

#Market Price Diffusion Model Parameters 
M0 = 100
mu_m = 0.1
sigma_m = 0.12
rho_ms = 0.5
rho_mv = 0

#General Parameters Simulation
T = 1
nsteps = 250
# paths_list = [1000, 10000, 100000] #1000, 10000, 100000
# b_list = [1, 3, 5, 10] #1, 3, 5, 10
# K_list = [80, 100, 120] #80, 100, 120
np.random.seed(200000)
# volatility = ['stochastic', 'constant'] #'stochastic', 'constant'
# option_type = ['Call', 'Put'] #'Call', 'Put'

#Square root diffusion
def SRD_generate_paths(v0, kappa_v, theta_v, sigma_v, T, nsteps, I, rand,
                       row, cho_matrix) :
           
    dt = T / nsteps
    v = np.zeros((nsteps + 1, I), dtype=np.float)
    v[0] = v0
    v_p = np.zeros_like(v)
    v_p[0] = v0
    sdt = math.sqrt(dt)
    for t in range(1, nsteps + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        v_p[t] = (v_p[t - 1] + kappa_v *
           (theta_v - np.maximum(0, v_p[t - 1])) * dt +
           np.sqrt(np.maximum(0, v_p[t - 1])) * sigma_v * ran[row] * sdt)
        v[t] = np.maximum(0, v_p[t])
    return(v)

#Stock price diffusion
def H93_generate_paths(S0, mu_s, v, T, nsteps, I, rand, row, volatility,
                       cho_matrix):
   
    dt = T / nsteps
    S = np.zeros((nsteps + 1, I), dtype=np.float)
    S[0] = S0
    sdt = math.sqrt(dt)
    for t in range(1, nsteps + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        if volatility == 'stochastic':
            S[t] = S[t - 1] * (1 + mu_s * dt + np.sqrt(v[t - 1]) *
             ran[row] * sdt)
        elif volatility == 'constant':
            S[t] = S[t - 1] * (1 + mu_s * dt + np.sqrt(v0) *
             ran[row] * sdt)
    return(S)
        
#Market price diffusion
def Mkt_generate_paths(M0, mu_m, sigma_m, T, nsteps, I, rand, row, cho_matrix):
   
    dt = T / nsteps
    M = np.zeros((nsteps + 1, I), dtype=np.float)
    M[0] = M0
    sdt = math.sqrt(dt)
    for t in range(1, nsteps + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        M[t] = M[t - 1] * (1 + mu_m * dt + sigma_m * ran[row] * sdt)
    return(M)
    
#Random number generator
def random_number_generator(nsteps, I):
    rand = np.random.standard_normal((3, nsteps + 1, I))
    return(rand)


#VALUATION
#=========

t0 = time()
    
results = pd.DataFrame()


if __name__ == '__main__':
    
    # Correlation Matrix
    covariance_matrix = np.zeros((3, 3), dtype=np.float)
    covariance_matrix[0] = [1.0, rho_vs, rho_ms]
    covariance_matrix[1] = [rho_vs, 1.0, rho_mv]
    covariance_matrix[2] = [rho_ms, rho_mv, 1.0]
    cho_matrix = np.linalg.cholesky(covariance_matrix)

    def final_result(option_type,paths_list, b_list, K_list, volatility,sigma_v):  
        results = pd.DataFrame()      
        for alpha in it.product(option_type,paths_list, b_list, K_list, volatility):
            print('\n', alpha, '\n')
            opt,I, b, K, vol= alpha
            
            #memory clean-up
            v, S, M, rand, st_dev = 0.0, 0.0, 0.0, 0.0, 0.0
            C_bar, R_bar, pi, r_m, r_mi = 0.0, 0.0, 0.0, 0.0, 0.0
      
            #random numbers
            rand = random_number_generator(nsteps, I)
            #volatility paths
            v = SRD_generate_paths(v0, kappa_v, theta_v, sigma_v, T,
                                  nsteps, I, rand, 1, cho_matrix)
            #stock price paths
            S = H93_generate_paths(S0, mu_s, v, T, nsteps, I, rand, 0,
                                   vol, cho_matrix)
            #market price paths
            M = Mkt_generate_paths(M0, mu_m, sigma_m, T, nsteps, I, rand, 2,
                                   cho_matrix)
            
            #market return over [0,T]
            r_m = M[nsteps, :] / M[0, :] - 1
            #MC estimator
            r_mi = (1 + r_m)**(1 - b)
            R_bar = np.sum(r_mi) / I
            if opt == 'Call':
                pi = np.maximum(S[-1] - K, 0) / (1 + r_m)**b
            elif opt == 'Put':
                pi = np.maximum(-S[-1] + K, 0) / (1 + r_m)**b
            C0 = np.sum(pi) / np.sum(r_mi)
            #standard deviation
            C_bar = np.sum(pi) / I
            st_dev = (1 / (I * R_bar**2)) * np.sum((pi - r_mi *
                           C_bar / R_bar)**2)
            st_dev = np.sqrt(st_dev) / np.sqrt(I)
            #Confidence intervals
            C0_inf = C0 - norm.ppf(0.05/2) * st_dev
            C0_sup = C0 + norm.ppf(0.05/2) * st_dev
           
            res = pd.DataFrame({'paths': I, 'strike': K,
                                'option value': C0,'stdev': st_dev,
                                'IC inf': C0_inf,'IC sup': C0_sup,
                                'relative risk aversion' : b, 'option type':
                                    opt, 'volatility': vol}, index= [0,])
        
            results = results.append(res, ignore_index=True)
        return(results)


#  Create function to save result to working directory
    def save_result(res, file_name):
        res.to_csv(os.path.join(path, file_name), header=True, index=None, sep=';')


# RUN and SAVE RESULTS
    table1 =  final_result(['Call'],[1000, 10000,100000],[5],[80,100,120],['stochastic'],[0.05])
    table2 =  final_result(['Call','Put'],[100000],[1,3,5,10],[80,100,120],['stochastic'],[0.05])
    table3 =  final_result(['Call','Put'],[100000],[3,5],[80,100,120],['constant','stochastic'],[0.2])

    save_result(table1, 'Table 1.csv')
    save_result(table2, 'Table 2.csv')
    save_result(table3, 'Table 3.csv')
    

    
    
    






    
    

