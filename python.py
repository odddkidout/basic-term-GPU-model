
import jax.numpy as np
from jax import jit,lax

"""Qx is mortality rate of the population"""
q = np.array([0.001,0.002,0.003,0.003,0.004,0.004,0.005,0.007,0.009,0.011])

"""Wx is Surrender rate of the population"""
w = np.array([0.05,0.07,0.08,0.10,0.14,0.20,0.20,0.20,0.10,0.04])

"""Yield curve"""
Yields = np.array([0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02])


""" We define some MPFs for policy information like premium, sum assured, Term , smoker status. this can be modified if an actuary wants to add more features"""
MPFs  = np.array(([10,25000,5,1],[100,25000,10,0],[80,20000,10,0],[80,20000,10,1]))


@jit
def num_in_force(x):
    numforce = np.array([1,0,0,0,0,0,0,0,0,0],dtype=np.float64)
    def b(i,a):
        return a.at[i].set((a[i-1]*(1-q[i-1]-w[i-1])))
    return lax.fori_loop(1,x[2],b,numforce)

@jit
def calculate_premium(x,numforce):
    premium = np.array([0,0,0,0,0,0,0,0,0,0],dtype=np.float64)
    def b(i,a):
        return a.at[i].set((x[0]*(numforce[i])))
    return lax.fori_loop(0,x[2],b,premium)
    
@jit
def calculate_claims(x,numforce):
    sum_assured = np.array([0,0,0,0,0,0,0,0,0,0],dtype=np.float64)
    def b(i,a):
        return a.at[i].set((x[1]*(numforce[i])*q[i]))
    return lax.fori_loop(0,x[2],b,sum_assured)

@jit
def calculate_discount(x,numforce):
    discount = np.array([1,0,0,0,0,0,0,0,0,0],dtype=np.float64)
    def b(i,a):
        return a.at[i].set((a[i-1]/(1+Yields[i-1])))
    return lax.fori_loop(1,x[2],b,discount)
        
        

for x in MPFs:
    numforce = num_in_force(x)
    premium = calculate_premium(x,numforce)
    claims = calculate_claims(x,numforce)
    disc_fact = calculate_discount(x,numforce)
    
    print("PV of contract")
    print(np.sum(np.multiply(disc_fact,np.subtract(premium,claims))))
    
