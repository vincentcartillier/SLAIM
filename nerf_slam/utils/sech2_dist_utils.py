import numpy as np

r"""
Some tools to handle the custom distribution build upon the sech2 function
- We can get the integral of that distribution (to normalize it to 1.))
- We can get the integral coefs needed to computed the expectation
"""


def sech(x):
    return 2 / (np.exp(x) + np.exp(-x))

def get_nerf_density(x, depth, sig, scale):
    if abs(x) > (depth+10*sig): return 0
    else: return scale * sech((x-depth)/sig)**2

#ray termination distribution
def custom_dist(x,u,sig,scale):
    return np.exp(-scale*sig* (np.tanh((x-u)/sig) - np.tanh(-u/sig)) )*get_nerf_density(x,u,sig,scale)


def compute_dist_norm(scale, sig):
    target_depth = 0.5
    x = np.linspace(target_depth-100*sig,target_depth+100*sig,100000)
    dt = x[1:] - x[:-1]
    dtm = np.mean(dt)
    dt = np.append(dt, dtm)

    d = np.array([custom_dist(e, target_depth, sig, scale) for e in x])

    I = np.sum(d*dt) #integral estimation

    return I



def f1(x,scale,sig):
    return x*np.exp(-scale*sig*(np.tanh(x)+1))*sech(x)**2

def f2(x,scale,sig):
    return np.exp(-scale*sig*(np.tanh(x)+1))*sech(x)**2

def get_integrals_coefs(scale, sig):
    x = np.linspace(-10,10,10000000)
    dt = x[1:] - x[:-1]
    dtm = np.mean(dt)
    dt = np.append(dt, dtm)

    y = f1(x, scale, sig)
    int_f1 = np.sum(y*dt)

    y = f2(x, scale, sig)
    int_f2 = np.sum(y*dt)

    return int_f1, int_f2


def get_dist_sum(scale, sigma):
    d = 0.5
    x = np.linspace(0,np.sqrt(3),1024)
    dist = np.array([custom_dist(e,d,sigma,scale) for e in x])
    sum_dist = np.sum(dist)
    return sum_dist
