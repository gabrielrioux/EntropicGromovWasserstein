# README  

## Overview  
This repository contains implementations of the algorithms in the paper:  
**Rioux G., Goldfeld Z., and Kato K.**  
*Entropic Gromov-Wasserstein Distances: Stability and Algorithms*, 2024.  

The code provides implementations of the accelerated gradient methods (Algorithms 1 and 2) for computing entropic Gromov-Wasserstein distances using the quadratic or inner product cost functions.  

---

## Repository Structure  
- **`EGWSolvers.py`**: Contains the implementation of the solvers.  
- **`example.py`**: Demonstrates the usage of the solvers with example inputs and outputs.  

---

## Installation  
To use the code, clone this repository and ensure you have [numpy](https://numpy.org/doc/stable/index.html) and [POT](https://pythonot.github.io/index.html) installed.

---

## Usage 

```python
from EGWSolvers import *

Nx,dx = 1000,2 # number of points, dimension of first marginal
Ny,dy = 1000,6 # number of points, dimension of second marginal
reg = 1 # regularization parameter
rng = np.random.default_rng(12345) 
 
x = rng.normal(0,.1,(Nx,dx)) # generate random support points of first marginal
y = rng.normal(0,.11,(Ny,dy)) # generate random support points of second marginal
wtx = rng.random(Nx) # generate random weights of first marginal
wtx/=np.sum(wtx) # normalize
wty = rng.random(Ny) # generate random weights of second marginal
wty/=np.sum(wty) # normalize

# center
x = x - mean(x,wtx)
y = y - mean(y,wty)

# compute cost and plan for EGW, can choose "quad" for quadratic or "inner" for inner product cost
# EGWAuto checks if a sufficient condition for convexity is met and uses the appropriate solver
# EGWConvex assumes the objective is convex
# EGWAdaptive does not assume convexity of the objective

#Optional arguments:   A (initial matrix dx x dy),
#                      L (Lipschitz constant of gradient),
#                      center (whether to center the distributions),
#                      delta (tolerance for termination based on norm of gradient),
#                      sinkhornOpts (optional arguments to pass to the POT sinkhorn method),
cost,plan = EGWAuto(x,y,wtx,wty,reg,"quad",A = None,L = None,center = True,delta = 1e-6,sinkhornOpts = {}) 
