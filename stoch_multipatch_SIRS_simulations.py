#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:39:51 2023

@author: albertakuno
"""
#%%
import numpy as np
from odeintw import odeintw
import openturns as ot
import matplotlib.pyplot as plt
import sympy as sm
import random
from scipy import linalg
from scipy.stats import norm
from statistics import NormalDist

#%%
"the system to be solved numerically"
def systemSIR(M, t, Beta, gamma):
    S, I, R = M
    #Nbarr=S+I+R
    dS_dt = Lambda -np.diag(S) @ Beta @ np.linalg.inv(np.diag(Nbar))@I\
     - np.diag(mu)@S + np.diag(tau)@R
    dI_dt = np.diag(S) @ Beta @ np.linalg.inv(np.diag(Nbar))@I \
    -np.diag(gamma+psi+mu)@I 
    dR_dt = np.diag(gamma)@I - np.diag(tau+mu)@R
    return np.array([dS_dt, dI_dt, dR_dt])

#%%

"declaring the parameters and the intial conditions"
gamma=np.array([1/14,1/14]);
Beta=np.array([[0.5,0.1],[0.5,0.1]])
Lambda=np.array([10,10])
#mu=np.array([0.6/(1000*365), 0.6/(1000*365)])
mu=np.array([1/30,1/70])
tau=np.array([1/180,1/180])
psi=np.array([0.0000,0.0000])
#Nbar = np.array([500,300])
Nbar=Lambda * (1/mu)
K=Nbar[0]+Nbar[1]

#Lambda = Nbar * mu
#K=4000
t = np.linspace(0, 200, 200)

#E_initial = np.array([50,60])

I_initial = np.array([0,1])

S_initial = np.array([Nbar[0]-I_initial[0],Nbar[1]-I_initial[1]])

R_initial = np.array([0,0])

M_initial = np.array([S_initial, I_initial, R_initial])

#Gamma1=Gamma[0]/mu[0]; Gamma2=Gamma[1]/mu[1]
#%%

"""plot of the solution of the ODE"""
sol=odeintw(systemSIR, M_initial, t, args=(Beta, gamma))
Sout = sol[:, 0, :]
Iout = sol[:, 1, :]
Rout = sol[:, 2, :]
S1=Sout[:,0]
S2=Sout[:,1]
I1=Iout[:,0]
I2=Iout[:,1]
R1=Rout[:,0]
R2=Rout[:,1]

fig, ax = plt.subplots(1, 2, constrained_layout=True)
ax[0].plot(t,S1,label=r"$S_{1}$",color="black")
ax[0].plot(t,I1,label=r"$I_{1}$",color="red")
ax[0].plot(t,R1,label=r"$R_{1}$",color="blue")
ax[0].legend(loc="best")
ax[0].set_title(r"Patch 1 trajectories")
#plt.ylim(0,45)
ax[0].set_ylabel("counts")
ax[0].grid(True)
ax[0].set_xlabel("time")

ax[1].plot(t,S2,label=r"$S_{2}$",color="black")
ax[1].plot(t,I2,label=r"$I_{2}$",color="red")
ax[1].plot(t,R2,label=r"$R_{2}$",color="blue")
ax[1].legend(loc="best")
ax[1].set_title(r"Patch 2 trajectories")
ax[1].set_ylabel("counts")
ax[1].grid(True)
ax[1].set_xlabel("time")
plt.show()
#%%

"""compute local patch R0"""
R01=Beta[0,0]/(gamma[0]+psi[0]+mu[0])
R02=Beta[1,1]/(gamma[1]+psi[1]+mu[1])
print("Patch 1 R0:", R01)
print("Patch 2 R0:", R02)      

R01_tld=Beta[1,0]/(gamma[0]+mu[0]); R02_tld=Beta[0,1]/(gamma[1]+mu[1])

(R01 - 1) * (R02 - 1) > R01_tld * R02_tld
#%%
"""compute global R0"""
Nbar=Lambda * 1/mu
F=np.diag(Nbar) @ Beta @ np.linalg.inv(np.diag(Nbar))
V=np.diag(gamma+mu+psi)
FV_inv=F @ np.linalg.inv(V)
eigvals=np.linalg.eigvals(FV_inv)
R_0=np.max(eigvals)
print("Global R0:",R_0)

#%%
"""Compute the equilibrium points (when R>1)
   brute force: iterate through possibility space (r)"""

s1, i1, r1, s2, i2, r2 = sm.symbols('s1, i1, r1, s2, i2, r2', negative=False, real = True)

S_1 = Lambda[0]/K - K*(mu[0]/Lambda[0])*Beta[0,0]*s1*i1 - K*(mu[1]/Lambda[1])*Beta[0,1]*s1*i2 - mu[0]*s1 + tau[0]*r1

I_1 = K*(mu[0]/Lambda[0])*Beta[0,0]*s1*i1 + K*(mu[1]/Lambda[1])*Beta[0,1]*s1*i2 - (gamma[0] + mu[0])*i1

R_1 = gamma[0]*i1 - (tau[0] + mu[0])*r1

S_2 = Lambda[1]/K - K*(mu[0]/Lambda[0])*Beta[1,0]*s2*i1 - K*(mu[1]/Lambda[1])*Beta[1,1]*s2*i2 - mu[1]*s2 + tau[1]*r2

I_2 = K*(mu[0]/Lambda[0])*Beta[1,0]*s2*i1 + K*(mu[1]/Lambda[1])*Beta[1,1]*s2*i2 - (gamma[1] + mu[1])*i2

R_2 = gamma[1]*i2 - (tau[1] + mu[1])*r2


""" use sympy's way of setting equations to zero"""
S_1Equal = sm.Eq(S_1, 0);
I_1Equal = sm.Eq(I_1, 0);
R_1Equal = sm.Eq(R_1, 0);
S_2Equal = sm.Eq(S_2, 0);
I_2Equal = sm.Eq(I_2, 0);
R_2Equal = sm.Eq(R_2, 0);

"""compute fixed points"""
equilibria =sm.solve( (S_1Equal, I_1Equal, R_1Equal, S_2Equal, I_2Equal, R_2Equal ), s1, i1, r1, s2, i2, r2,check=False,force=True)
equilibria

#%%

"""Filter the equilibrium to pick only those points with positive values"""
filtered_equilibria = []
for ss in equilibria:
    check_repos = all([sm.re(s) >= 0 for s in ss])
    if (check_repos):
        filtered_equilibria.append(tuple((sm.re(s) for s in ss)))

print(filtered_equilibria)
#%%

"""Obtain the Jacobian of the deterministic model and check if the equilibrium points
   obtained above are stable or not (based on their eigenvalues)"""

eqMat = sm.Matrix([ S_1, I_1, R_1, S_2, I_2, R_2 ])
Mat = sm.Matrix([ s1, i1, r1, s2, i2, r2 ])
jacMat = eqMat.jacobian(Mat)
print('Jacobian %s' % jacMat)
print('--------------------------------------------------------------------------------------------')


""" iterate through list of equilibria """
for item in filtered_equilibria:
    print(item)
    eqmat = jacMat.subs([ (s1, item[0]), (i1, item[1]), (r1, item[2]), (s2, item[3]), (i2, item[4]),  (r2, item[5]) ])
    evals = list(eqmat.eigenvals().keys())
    print(f'The eigenvalues for the fixed point (%s, %s, %s, %s, %s, %s) are {evals}:' 
        %(item[0], item[1], item[2], item[3], item[4], item[5]))
    print('--------------------------------------------------------------------------------------------')

#%%

"""select the endemic equilibrium (EE) point and evaluate the Jacobian at these points (with the states arranged as S1,I1,R1,S2,I2,R2)"""


EE = [i for i in filtered_equilibria if i[1] != 0][0]
#EE=list_EE[0]
JacEE=jacMat.subs([ (s1,EE[0]), (i1,EE[1]), (r1,EE[2]), (s2,EE[3]), (i2,EE[4]), (r2,EE[5]) ])
#print(JacEE)

#%%
"""Here, I obtain the B matrix but now with the states arranged as S1,S2,I1,I2,R1,R2"""

s1, s2, i1, i2, r1, r2 = sm.symbols('s1, s2, i1, i2, r1, r2', negative=False, real = True)

B11 = -K*(mu[0]/Lambda[0])*Beta[0,0]*i1 - K*(mu[1]/Lambda[1])*Beta[0,1]*i2 - mu[0]
B13 = -K*(mu[0]/Lambda[0])*Beta[0,0]*s1
B14 = - K*(mu[1]/Lambda[1])*Beta[0,1]*s1
B22 = -K*(mu[0]/Lambda[0])*Beta[1,0]*i1 - K*(mu[1]/Lambda[1])*Beta[1,1]*i2 - mu[1]
B23 = -K*(mu[0]/Lambda[0])*Beta[1,0]*s2
B24 = -K*(mu[1]/Lambda[1])*Beta[1,1]*s2
B31 = K*(mu[0]/Lambda[0])*Beta[0,0]*i1 + K*(mu[1]/Lambda[1])*Beta[0,1]*i2
B33 = K*(mu[0]/Lambda[0])*Beta[0,0]*s1 - (gamma[0] + mu[0]) 
B34 = K*(mu[1]/Lambda[1])*Beta[0,1]*s1
B42 = K*(mu[0]/Lambda[0])*Beta[1,0]*i1 + K*(mu[1]/Lambda[1])*Beta[1,1]*i2
B43 = K*(mu[0]/Lambda[0])*Beta[1,0]*s2 
B44 = K*(mu[1]/Lambda[1])*Beta[1,1]*s2 - (gamma[1] + mu[1])
Bmat = sm.Matrix([ [B11, 0, B13, B14, tau[0], 0],[0, B22, B23, B24, 0, tau[1]],[B31, 0, B33, B34, 0, 0 ],[0, B42, B43, B44, 0, 0],[0, 0, gamma[0], 0, -(tau[0]+mu[0]), 0],[0, 0, 0, gamma[1], 0, -(tau[1] + mu[1])] ]  )
Bmatev_EE=Bmat.subs([ (s1,EE[0]), (s2,EE[3]), (i1,EE[1]), (i2,EE[4]), (r1,EE[2]), (r2,EE[5])])
print(Bmatev_EE)

#%%

"""Here, I obtain the covariance matrix C, but now with the states arranged as S1,S2,I1,I2,R1,R2"""
CC11 = Lambda[0]/K +mu[0]*s1 + tau[0]*r1 + K*(mu[0]/Lambda[0])*Beta[0,0]*s1*i1 + K*(mu[1]/Lambda[1])*Beta[0,1]*s1*i2
CC13 = CC31 = -K*(mu[0]/Lambda[0])*Beta[0,0]*s1*i1 - K*(mu[1]/Lambda[1])*Beta[0,1]*s1*i2
CC22 = Lambda[1]/K + mu[1]*s2 + tau[1]*r2 + K*(mu[0]/Lambda[0])*Beta[1,0]*s2*i1 + K*(mu[1]/Lambda[1])*Beta[1,1]*s2*i2
CC24 = CC42 = -K*(mu[0]/Lambda[0])*Beta[1,0]*s2*i1 - K*(mu[1]/Lambda[1])*Beta[1,1]*s2*i2
CC33 = K*(mu[0]/Lambda[0])*Beta[0,0]*s1*i1 + K*(mu[1]/Lambda[1])*Beta[0,1]*s1*i2 + (gamma[0]+mu[0])*i1
CC44 = K*(mu[0]/Lambda[0])*Beta[1,0]*s2*i1 + K*(mu[1]/Lambda[1])*Beta[1,1]*s2*i2 + (mu[1]+ gamma[1])*i2
CC55 = (mu[0] + tau[0])*r1 + gamma[0]*i1
CC66 = (mu[1] + tau[1])*r2 + gamma[1]*i2
CCmat = sm.Matrix( [ [CC11, 0, CC13, 0, -tau[0]*r1, 0], [0, CC22, 0, CC24, 0, -tau[1]*r2], [CC31, 0, CC33, 0, -gamma[0]*i1, 0], [0, CC42, 0, CC44, 0, -gamma[1]*i2], [-tau[0]*r1, 0, -gamma[0]*i1, 0, CC55, 0], [0, -tau[1]*r2, 0, -gamma[1]*i2, 0, CC66] ] )
CCmatev_EE=CCmat.subs([ (s1,EE[0]), (s2,EE[3]), (i1,EE[1]), (i2,EE[4]), (r1,EE[2]), (r2,EE[5])])
print(CCmatev_EE)

#%%
"""in this cell, we solve the lyapunov function "B@Sigma + Sigma@B^{T}=-CC^{T}" to obtain Sigma"""
Bmat_Ar=np.array(Bmatev_EE).astype(np.float64)
CCmat_Ar=np.array(CCmatev_EE).astype(np.float64)
Sigma = linalg.solve_continuous_lyapunov(Bmat_Ar, -CCmat_Ar)
print(Sigma)

#%%
"""check if the solution is correct"""
np.allclose(Bmat_Ar.dot(Sigma) + Sigma.dot(Bmat_Ar.T), -CCmat_Ar) 

#%%

"""Here, i pick the endemic points correponding to the infected compartments with their corresponding variance covariance matrix"""
Ivec_bar=np.array([EE[1],EE[4]]).astype(np.float64)
Ivec_Cov=np.array([[Sigma[2,2],Sigma[2,3]],[Sigma[3,2],Sigma[3,3]]])
print("Normalized endemic infection levels:",Ivec_bar)
print("Normalized Cov mat of infected:")
print(Ivec_Cov)

#%%

Ivec_bar_K = Ivec_bar*K
Ivec_Cov_K = K*Ivec_Cov

print("Non_normalized endemic infection levels:",Ivec_bar_K)
print("Non normalized Cov mat of infected:")
print(Ivec_Cov_K)


#%%


"""Finally, here I estimate and compute the probability and Expected time to extinction (ETTE) using the multivariate normal distribution"""
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import sqrtm
import statsmodels.stats.correlation_tools as sm
#from scipy import stats
e1 = np.array([1,0])
e2 = np.array([0,1])
#eta_vec1 = np.array([3/5, 4/5]) #direction
#eta_vec2 = np.array([3/5, 4/5])
#alpha_vec = np.array([0.2,0.5]) 

muI = Ivec_bar_K
covI = Ivec_Cov_K
#vec05 = np.array([0.5,0.5])

x = sqrtm(np.linalg.inv(covI)) @ (e1 - muI)
y = sqrtm(np.linalg.inv(covI)) @ (e2 - muI)

I = np.eye(2)
zero_mu = np.array([0,0])

sqrt_det_Sigma = np.sqrt(np.linalg.det(covI))

multinormal = ot.Normal(muI, ot.CovarianceMatrix(covI))

#I1 = np.arange(1,351,1)
#I2 = np.arange(1,351,1)

prob_num1 = mvn.pdf(x, mean=zero_mu, cov=I)
prob_num2 = mvn.pdf(y,mean=zero_mu, cov=I)
#prob_num11 = stats.multivariate_normal.cdf(e1,mean=muI, cov=covI)
#prob_num22 = stats.multivariate_normal.cdf(e2,mean=muI, cov=covI)

prob_denom1 = multinormal.computeProbability(ot.Interval([0.5,0], [np.inf,1])) 
prob_denom2 = multinormal.computeProbability(ot.Interval([0,0.5], [1,np.inf])) 
prob_denom3 = multinormal.computeProbability(ot.Interval([0.5,0.5], [np.inf,np.inf])) 
prob_denom = prob_denom1 + prob_denom2 + prob_denom3




#q_dot_arr_dot = np.zeros(len(I1))
#for i in range(1,len(I1)):
#for i in range(1,3):
    #print(i)
#    e1_arr = np.array([i,0])
#    e2_arr = np.array([0,i])
#    x_arr = sqrtm(np.linalg.inv(covI)) @ (e1_arr - muI)
#    y_arr = sqrtm(np.linalg.inv(covI)) @ (e2_arr - muI)
#    prob_num1_arr = mvn.pdf(x_arr,mean=zero_mu, cov=I)
#    prob_num2_arr = mvn.pdf(y_arr,mean=zero_mu, cov=I)
#    print(prob_num1_arr)
#    print(prob_num2_arr)
    
 #   prob_denom1 = multinormal.computeProbability(ot.Interval([0.5,0], [np.inf,1])) 
 #   prob_denom2 = multinormal.computeProbability(ot.Interval([0,0.5], [1,np.inf])) 
#    prob_denom3 = multinormal.computeProbability(ot.Interval([0.5,0.5], [np.inf,np.inf])) 
#    prob_denom = prob_denom1 + prob_denom2 + prob_denom3
#    q_dot_arr_dot[i] = (prob_num1_arr + prob_num2_arr)/(sqrt_det_Sigma*prob_denom)

#plt.plot(I1,q_dot_arr_dot)

#prob_num11 = stats.multivariate_normal.cdf(e1,mean=muI, cov=covI)

#x = sqrtm(np.linalg.inv(covI)) @ (e1 - muI)
#y = sqrtm(np.linalg.inv(covI)) @ (e2 - muI)
#z = sqrtm(np.linalg.inv(covI)) @ (vec05 - muI)
#dist = mvn(mean=muI, cov=covI)
#a = eta_vec1.T @ sqrtm(np.linalg.inv(covI)) @ alpha_vec
#b = eta_vec2.T @ sqrtm(np.linalg.inv(covI)) @ alpha_vec

#prob = multinormal.computeProbability(ot.Interval([0.5,0.5], [np.inf,np.inf])) 
#print(prob)
#print("CDF:", dist.cdf(np.array([2,4])))
#Cov_corr = sm.cov_nearest(covariance, method='clipped', threshold=1e-15, n_fact=100, return_all=False)
#Cov_corr
#p_num = mvn.cdf(z)

#print(p_num)
#print(p_denom)

q_dot1_dot = (prob_num1 + prob_num2)/(sqrt_det_Sigma*prob_denom)



g_hat = ((mu[0] + gamma[0]) * prob_num1 + (mu[1] + gamma[1]) * prob_num2)/(sqrt_det_Sigma*prob_denom) # from multivariate pdf
#g_hat2 = ((mu[0] + gamma[0]) * prob_num11 + (mu[1] + gamma[1]) * prob_num22)/(prob_denom) #from multivariate cdf
print("q_hat: ",q_dot1_dot)
#print("g_hat_MV_CD",g_hat2)
ETTE = 1/g_hat #expected time to extinction 
#ETTE2 = 1/g_hat2 #expected time to extinction 
print("ETTE: ",ETTE)
#print("ETTE_MV_cdf",ETTE2)

#%%
def q_dot_1_dot(I1, I2, muI, covI):
    #u=np.array([0])
    #v=np.array([0])
    e1 = np.array([I1,0])
    e2 = np.array([0,I2])
    muI = Ivec_bar_K
    covI = Ivec_Cov_K
    x = sqrtm(np.linalg.inv(covI)) @ (e1 - muI)
    y = sqrtm(np.linalg.inv(covI)) @ (e2 - muI)
    I = np.eye(2)
    zero_mu = np.array([0,0])
    sqrt_det_Sigma = np.sqrt(np.linalg.det(covI))
    multinormal = ot.Normal(muI, ot.CovarianceMatrix(covI))
    prob_num1 = mvn.pdf(x, mean=zero_mu, cov=I)
    prob_num2 = mvn.pdf(y,mean=zero_mu, cov=I)
    prob_denom1 = multinormal.computeProbability(ot.Interval([0.5,0], [np.inf,1])) 
    prob_denom2 = multinormal.computeProbability(ot.Interval([0,0.5], [1,np.inf])) 
    prob_denom3 = multinormal.computeProbability(ot.Interval([0.5,0.5], [np.inf,np.inf])) 
    prob_denom = prob_denom1 + prob_denom2 + prob_denom3
    q_dot1_dot = (prob_num1 + prob_num2)/(sqrt_det_Sigma*prob_denom)
    return q_dot1_dot

m=100
q =np.arange(1, 3, 1)
r = np.arange(1, 3, 1)
X,Y = np.meshgrid(q, q)
Q=q_dot_1_dot(X, Y, muI, covI)
plt.contour(X, Y, Q, colors='red');
#plt.scatter(x1, x2,alpha=0.3,cmap='viridis')
#plt.plot(x1,x2, color='gray', marker='+')
plt.xlabel(r'$I_{1}$') 
# naming the y axis 
plt.ylabel(r'$I_{2}$') 
# giving a title to my graph 
plt.title(r'Marginal distribution of patch 1 and patch 2 infected') 

#%%

"""Using univariate normal distribution to approximate distribution and expected time to extinction.
   Just trying this because using the multivariate distribution is not give close approximations to
   exact (from simulation) value
"""


mu_sum_I1_I2 = np.sum(Ivec_bar_K) 

var_sum_I1_I2 = np.sum(np.diag(Ivec_Cov_K)) + 2*Ivec_Cov_K[0,1]

sigma_sum_I1_I2 = np.sqrt(var_sum_I1_I2)

u = (1-mu_sum_I1_I2)/sigma_sum_I1_I2
v = (mu_sum_I1_I2-0.5)/sigma_sum_I1_I2

norm_pdf = norm.pdf(u , loc = 0 , scale = 1 )
norm_cdf = NormalDist(mu=0, sigma=1).cdf(v)

#norm_pdf = psi(x)
#norm_cdf = phi(y)


marg_dist_sum_I1_I2 = norm_pdf/(sigma_sum_I1_I2 * norm_cdf )


g_hat_sum_I1_I2 = ((mu[0] + gamma[0]) + (mu[1] + gamma[1]))*marg_dist_sum_I1_I2 # from multivariate pdf

print("Approx. marginal dist. from univariate dist:", marg_dist_sum_I1_I2)

print("Approx. ETTE from univariate dist:", 1/g_hat_sum_I1_I2)

#%%

def Gillespie_Multipatch2_SIRS(S1, I1, R1, S2, I2, R2, params, tend):

        Lambda1 = params[0]
        Lambda2 = params[1]
        mu1 = params[2]
        mu2 = params[3]
        psi1 = params[4]
        psi2 = params[5]
        tau1 = params[6]
        tau2 = params[7]
        beta11 = params[8]
        beta12 = params[9]
        beta21 = params[10]
        beta22 = params[11]
        gamma1 = params[12]
        gamma2 = params[13]
        
        
        

        t = 0
        X_t = []
        flag = True
        firstI1 = 1
        firstI2 = 1
        

        while t < tend and (S1 + I1 + S2 + I2 >= 1) and flag:
                
                N1 = S1 + I1 + R1
                N2 = S2 + I2 + R2
                
                X_t.append([t,S1,I1,R1,S2,I2,R2])
                
                
                rates = [Lambda1, Lambda2, mu1*S1, mu2*S2, (mu1 + psi1)*I1, (mu2 + psi2)*I2,\
                         mu1*R1, mu2*R2, tau1*R1, tau2*R2, beta11*S1*I1/N1 + beta12*S1*I2/N2,\
                         beta21*S2*I1/N1 + beta22*S2*I2/N2, gamma1*I1, gamma2*I2]
                
                rate_sum = sum(rates)
                if rate_sum==0:
                    t=tend
                    break
                if I1 <= 0: 
                    if firstI1 == 1:
                       firstI1 = 2
                    #if firstI2 == 1:
                     #  firstI2 = 2
                    else:
                        flag = False
                if I2 <= 0: 
                    if firstI2 == 1:
                       firstI2 = 2
                    #if firstI2 == 1:
                     #  firstI2 = 2
                    else:
                        flag = False
                #if I<=0: 
                  #  flag == False
                
                    #rate_sum = 1e-10
                

                tau = np.random.exponential(scale=1/rate_sum)

                t += tau

                rand = random.uniform(0,1)


                # Immigration of Susceptibles in Patch 1
                if rand * rate_sum <= rates[0]:
                    S1 += 1
                    I1 = I1
                    R1 = R1
                    S2 = S2
                    I2 = I2
                    R2 = R2

                # Immigration of Susceptibles in Patch 2
                elif rand * rate_sum > rates[0] and rand * rate_sum <= sum(rates[:2]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 = R1
                    S2 += 1
                    I2 = I2
                    R2 = R2
                    
                # Death of Susceptibles in Patch 1
                elif rand * rate_sum > sum(rates[:2]) and rand * rate_sum <= sum(rates[:3]):
                #else:
                    S1 -= 1
                    I1 = I1
                    R1 = R1
                    S2 = S2
                    I2 = I2
                    R2 = R2
                
                # Death of Susceptibles in Patch 2
                elif rand * rate_sum > sum(rates[:3]) and rand * rate_sum <= sum(rates[:4]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 = R1
                    S2 -= 1
                    I2 = I2
                    R2 = R2
                
                # Death of infected in Patch 1
                elif rand * rate_sum > sum(rates[:4]) and rand * rate_sum <= sum(rates[:5]):
                #else:
                    S1 = S1
                    I1 -= 1
                    R1 = R1
                    S2 = S2
                    I2 = I2
                    R2 = R2
                    
                # Death of infected in Patch 2
                elif rand * rate_sum > sum(rates[:5]) and rand * rate_sum <= sum(rates[:6]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 = R1
                    S2 = S2
                    I2 -= 1
                    R2 = R2
                
                # Death of rcovered in Patch 1
                elif rand * rate_sum > sum(rates[:6]) and rand * rate_sum <= sum(rates[:7]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 -= 1
                    S2 = S2
                    I2 = I2
                    R2 = R2
                    
                # Death of rcovered in Patch 2
                elif rand * rate_sum > sum(rates[:7]) and rand * rate_sum <= sum(rates[:8]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 = R1
                    S2 = S2
                    I2 = I2
                    R2 -= 1
                    
                # Loss of immunity in Patch 1
                elif rand * rate_sum > sum(rates[:8]) and rand * rate_sum <= sum(rates[:9]):
                #else:
                    S1 += 1
                    I1 = I1
                    R1 -= 1
                    S2 = S2
                    I2 = I2
                    R2 = R2
                    
                # Loss of immunity in Patch 2
                elif rand * rate_sum > sum(rates[:9]) and rand * rate_sum <= sum(rates[:10]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 = R1
                    S2 += 1
                    I2 = I2
                    R2 -= 1
                    
                # Infection of susceptible in Patch 1
                elif rand * rate_sum > sum(rates[:10]) and rand * rate_sum <= sum(rates[:11]):
                #else:
                    S1 -= 1
                    I1 += 1
                    R1 = R1
                    S2 = S2
                    I2 = I2
                    R2 = R2
                    
                # Infection of susceptible in Patch 2
                elif rand * rate_sum > sum(rates[:11]) and rand * rate_sum <= sum(rates[:12]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 = R1
                    S2 -= 1
                    I2 += 1
                    R2 = R2
                    
                # Recovery of infected in Patch 1
                elif rand * rate_sum > sum(rates[:12]) and rand * rate_sum <= sum(rates[:13]):
                #else:
                    S1 = S1
                    I1 -= 1
                    R1 += 1
                    S2 = S2
                    I2 = I2
                    R2 = R2
                    
                # Recovery of infected in Patch 2
                elif rand * rate_sum > sum(rates[:13]):# and rand * rate_sum <= sum(rates[:14]):
                #else:
                    S1 = S1
                    I1 = I1
                    R1 = R1
                    S2 = S2
                    I2 -= 1
                    R2 += 1
                    
        X_t.append([t,S1,I1,R1,S2,I2,R2])

        return (np.array(X_t).transpose())
#%%

Lambda1 = Lambda[0]
Lambda2 = Lambda[1]
mu1 = mu[0]
mu2 = mu[1]
psi1 = psi[0]
psi2 = psi[1]
tau1 = tau[0]
tau2 = tau[1]
beta11 = Beta[0,0]
beta12 = Beta[0,1]
beta21 = Beta[1,0]
beta22 = Beta[1,1]
gamma1 = gamma[0]
gamma2 = gamma[1]

params = [Lambda1, Lambda2, mu1, mu2, psi1, psi2, tau1, tau2, beta11, beta12, beta21, beta22, gamma1, gamma2]


I01 = int(EE[1]*K)
R01 = int(EE[2]*K)
S01 = int(EE[0]*K)

I02 = int(EE[4]*K)
R02 = int(EE[5]*K)
S02 = int(EE[3]*K)



#I01 = 9.1981 #EE[1]*K
#R01 = 9.5814 #EE[2]*K
#S01 = 181.2205 # EE[0]*K

#I02 = 26.8081 #EE[4]*K
#R02 = 26.1118 #EE[5]*K
#S02 = 247.0802 #EE[3]*K

#N0 = S01 + I01 + R01 + S02 + I02 + R02


#%% Run simulations

number_of_simulations=500  #100
tend = 5000

random.seed(1000)

sim_out_mult_SIRS = []

for q in range(number_of_simulations):    
    sim_out_mult_SIRS.append(Gillespie_Multipatch2_SIRS(S01, I01, R01, S02, I02, R02, params, tend))
    #plt.hist(sim_out[:,2], bins=10)
#%%


#plot patch 2 infected simulations (patch 1 infected)
fig, ax1 = plt.subplots()

for i in range(number_of_simulations):
    ax1.plot(sim_out_mult_SIRS[i][0],sim_out_mult_SIRS[i][2])
ax1.set_xlabel("time")
ax1.set_ylabel("count")
#ax1.set_title("Patch 1 infected", fontsize = 20, weight = "bold")
plt.savefig('/Users/albertakuno/Library/CloudStorage/Dropbox/Stochastic Epidemic models/model/figures/P1_infection_curves_varying_beta21_0.3_N1_500_N2_1500.png', format="png", dpi=500 )
#plt.legend()

fig, ax2 = plt.subplots()
for i in range(number_of_simulations):
    ax2.plot(sim_out_mult_SIRS[i][0],sim_out_mult_SIRS[i][5])
ax2.set_xlabel("time")
ax2.set_ylabel("count")
#ax2.set_title("Patch 1 infected", fontsize = 20, weight = "bold")
plt.savefig('/Users/albertakuno/Library/CloudStorage/Dropbox/Stochastic Epidemic models/model/figures/P2_infection_curves_varying_beta21_0.3_N1_500_N2_1500.png', format="png", dpi=500 )
#plt.legend()

fig, ax3 = plt.subplots()
for i in range(number_of_simulations):
    ax3.plot(sim_out_mult_SIRS[i][0],sim_out_mult_SIRS[i][2]+sim_out_mult_SIRS[i][5])
ax3.set_xlabel("time")
ax3.set_ylabel("count")
#ax3.set_title(r"Global infected", fontsize = 20, weight = "bold")
plt.savefig('/Users/albertakuno/Library/CloudStorage/Dropbox/Stochastic Epidemic models/model/figures/global_infection_curves_varying_beta21_0.3_N1_500_N2_1500.png', format="png", dpi=500 )
#plt.legend()


#%%

sim_out_long_arr_mult_SIRS = []

for i in range(number_of_simulations): 
    sim_out_long_arr_mult_SIRS.append((sim_out_mult_SIRS[i][2]+sim_out_mult_SIRS[i][5])[sim_out_mult_SIRS[i][0]>=0.1])


#%%

#plt.hist(sim_out_long_arr)

flat_list_mult_SIRS = [item for sublist in sim_out_long_arr_mult_SIRS for item in sublist]
#flat_list = np.concatenate([x.ravel() for x in sim_out_long_arr]).tolist()
#flat_list = loaded_sim_out_long_arr.flatten()


#%%


#plt.hist(flat_list_mult_SIRS, bins=range(min(flat_list_mult_SIRS).astype(int),max(flat_list_mult_SIRS).astype(int)), density=True)

#%% 

flat_list_mult_SIRS_remove_zeros = [i for i in flat_list_mult_SIRS if i!=0]

#%%

mean = np.mean(flat_list_mult_SIRS_remove_zeros)
std = np.std(flat_list_mult_SIRS_remove_zeros)

print("mean:", mean)
print("std:", std)


#%%
fig, ax1 = plt.subplots()
freq = plt.hist(flat_list_mult_SIRS_remove_zeros, bins=range(min(flat_list_mult_SIRS_remove_zeros).astype(int),max(flat_list_mult_SIRS_remove_zeros).astype(int)), density=True)
x = np.linspace(mu_sum_I1_I2- 3*sigma_sum_I1_I2, mu_sum_I1_I2 + 3*sigma_sum_I1_I2, 100)
plt.plot(x, norm.pdf(x, mu_sum_I1_I2, sigma_sum_I1_I2))
ax1.set_xlabel("Count")
ax1.set_ylabel("Distribution" )
plt.savefig('/Users/albertakuno/Library/CloudStorage/Dropbox/Stochastic Epidemic models/simulation_codes/Figures/hist_varyingbeta21_0.8_N1_500_N2_1500.png', format="png", dpi=300 )

q_dot_1_dot_1 = freq[0][0] #Marginal distribution of infected individuals conditioned on non extinction

print("Exact marginal dist. of one infected in multipatch at QS:", q_dot_1_dot_1)

print("Approx. marginal dist. of one infected in multipatch at QS:", q_dot1_dot)

#%%

index_TTE_mult_SIRS = [] #create a list to append the indices of the time to extinction (the list contains Nones for chains which did not hit zero)
unfiltered_TTE_mul_SIRS = [] #creat a list to append TTEs (this list will also contain full arrays of times at None positions)


for i in range(number_of_simulations):
    
    #append the indices of the time to extinction (where the sum of I1 and I2 (I1+I2) first hits 0), S1, S2. R1, R2 at time to extinction
    index_TTE_mult_SIRS.append( next((idx for idx, val in np.ndenumerate((sim_out_mult_SIRS[i][2]+sim_out_mult_SIRS[i][2])[sim_out_mult_SIRS[i][0]>=0.1]) if val==0),None))
    unfiltered_TTE_mul_SIRS.append( (sim_out_mult_SIRS[i][0])[next((idx for idx, val in np.ndenumerate((sim_out_mult_SIRS[i][2]+sim_out_mult_SIRS[i][2])[sim_out_mult_SIRS[i][0]>=0.1]) if val==0),None)] )
    
#obtain the None indices to help us identify and remove all the full array of times corresponding to
#osition of Nones in unfiltered_TTE
none_indices = [x  for x in range(0,len(index_TTE_mult_SIRS)) if index_TTE_mult_SIRS[x] is None]

def remove_by_indices(iter, idxs):
    """
    this function will remove the full arrays of 
    times correponding to the None positions in index_TTE
    
    """
    return [e for i, e in enumerate(iter) if i not in idxs]

array_TTE_mult_SIRS = remove_by_indices(unfiltered_TTE_mul_SIRS, none_indices) #obtain an array of the time to extinction (TTE) after removing the full array of times in None positions

#plt.hist(array_TTE,bins=50, density=True)

exact_q_dot_1_dot = q_dot_1_dot_1 #exact marginal distribution of 1 infected individual at quassi stationarity

exact_ETTE_mult_SIRS = np.mean(array_TTE_mult_SIRS) # exact (simulated) expected time to extinction as a mean of the simulated array of TTE

approx_q_dot_1 = q_dot1_dot #approximated marginal distribution of 1 infected individual at quassi stationarity

approx_ETTE_mult_SIRS = ETTE #approximated time to extinction of the disease starting at quassi stationarity


print("Approx. ETTE:", approx_ETTE_mult_SIRS)

print("Exact ETTE:", exact_ETTE_mult_SIRS)

#%%
