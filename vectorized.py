import jax.numpy as jnp

q = jnp.array([0.001,0.002,0.003,0.003,0.004,0.004,0.005,0.007,0.009,0.011])
w = jnp.array([0.05,0.07,0.08,0.10,0.14,0.20,0.20,0.20,0.10,0.04])
interest_rate = .02
face_value = 500_000
annual_premium = 3_000

# Policies in force starts at one, we add one to the beginning of the array
pols_if = jnp.concatenate([jnp.ones(1), jnp.cumprod((1-q)*(1-w))])
timesteps = jnp.arange(pols_if.shape[0])
print("This is the probability the policy is still in force at each timestep:", pols_if)
print("Timesteps:", timesteps)
# No cashflows out of the policy at time 0, add a zero to the beginning of the array
claims = pols_if * jnp.concatenate([jnp.zeros(1), q]) * face_value
premiums = pols_if * annual_premium

net_cashflows = (premiums - claims)
print("This is the net cashflows:", net_cashflows)
discount_factors = jnp.float_power(1 + interest_rate, -timesteps)
discounted_cashflows = net_cashflows * discount_factors
NPV = jnp.sum(discounted_cashflows)
print("NPV:", NPV)
