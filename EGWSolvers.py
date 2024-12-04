import numpy as np 
from ot.bregman import sinkhorn

#Compute mean of a discrete distribution on N points in R^d
#Input:     x: support points (Nxd)
#          wt: probability mass attributed to each of the N points (Nx1)
#Return: mean of distribution
def mean(x,wt):
    return np.dot(wt,x)

#Compute fourth moment of a discrete distribution on N points in R^d
#Input:     x: support points (Nxd)
#          wt: probability mass attributed to each of the N points (Nx1)
#Return: fourth moment of distribution
def fourthMoment(x,wt):
    return np.dot(np.einsum('ij,ij->i',x,x)**2,wt)


#Decomposes the quadratic cost matrix M_{ij}=\|x_i-x_j\|^2 as A_1A_2^T  
#Used to compute GW distance with square loss efficiently given a plan (if N_x>>d_x)
#Input:   X: design matrix for X (N_x,d_x)
#Return: matrices for decomposition A_1,A_2
#From Scetbon, Peyré, and Cuturi - Linear-Time Gromov Wasserstein distances using Low Rank Couplings and Costs
#https://proceedings.mlr.press/v162/scetbon22b.html
def factorizedSquareEuclidean(X):
    squareNorm = np.sum(X**2, axis=1)
    A_1 = np.zeros((np.shape(X)[0], 2 + np.shape(X)[1]))
    A_1[:, 0] = squareNorm
    A_1[:, 1] = np.ones(np.shape(X)[0])
    A_1[:, 2:] = -2 * X

    A_2 = np.zeros((2 + np.shape(X)[1], np.shape(X)[0]))
    A_2[0, :] = np.ones(np.shape(X)[0])
    A_2[1, :] = squareNorm
    A_2[2:, :] = X.T

    return A_1, A_2

#Compute the gradient of the objective $32\|A\|_F^2+OT_{A,reg}(mu,nu)$ for the cost $c_A(x,y)=-4\|x\|^2\|y\|^2-32x^{T}Ay$
#Input:     x: support points of mu (N_x x d_x)
#           y: support points of nu (N_y x d_y)
#         wtx: probability mass attributed to the N_x points of mu (N_x x 1)
#         wty: probability mass attributed to the N_y points of nu (N_y x 1)
#           A: matrix at which to compute gradient (d_x x d_y)
#constantCost: precomputed value of -4\|x\|^2\|y\|^2 (N_x x N_y)
#         reg: regularization parameter for entropic Gromov-Wasserstein
#sinkhornOpts: options to pass to Sinkhorn solver (dict)
#Return: gradient of objective corresponding to quadratic cost (d_x x d_y)
def gradientQuadratic(x,y,wtx,wty,A,constantCost,reg,sinkhornOpts):
    cost = constantCost - 32*np.dot(np.dot(x,A),y.T) 
    plan = sinkhorn(wtx,wty,cost,reg,**sinkhornOpts)
    return 64*A-32*np.dot(np.dot(x.T,plan),y),plan

#Compute value of -4\|x\|^2\|y\|^2
#Input:     x: support points of mu (N_x x d_x)
#           y: support points of nu (N_y x d_y)
#Return: -4\|x\|^2\|y\|^2 (N_x x N_y)
def computeConstantCost(x,y):
    return -4*np.outer(np.einsum('ij,ij->i',x,x),np.einsum('ij,ij->i',y,y))
    
#Compute the gradient of the objective 8\|A\|_F^2+OT_{A,reg}(mu,nu)$ for the cost $c_A(x,y)=-8x^{T}Ay$
#Input:     x: support points of mu (N_x x d_x)
#           y: support points of nu (N_y x d_y)
#         wtx: probability mass attributed to the N_x points of mu (N_x x 1)
#         wty: probability mass attributed to the N_y points of nu (N_y x 1) 
#           A: matrix at which to compute gradient (d_x x d_y)
#         reg: regularization parameter for entropic Gromov-Wasserstein
#sinkhornOpts: options to pass to Sinkhorn solver (dict)
#Return: gradient of objective corresponding to inner product cost (N_x x N_y)   
def gradientInnerProduct(x,y,wtx,wty,A,reg,sinkhornOpts):
    cost = - 8*np.dot(np.dot(x,A),y.T)
    plan = sinkhorn(wtx,wty,cost,reg,**sinkhornOpts)
    return 16*A-8*np.dot(np.dot(x.T,plan),y),plan

#Choose the right gradient (for quadratic or inner product cost) and return lambda function
#Input:     x: support points of mu (N_x x d_x)
#           y: support points of nu (N_y x d_y)
#         wtx: probability mass attributed to the N_x points of mu (N_x x 1)
#         wty: probability mass attributed to the N_y points of nu (N_y x 1) 
#         reg: regularization parameter for entropic Gromov-Wasserstein
#        cost: quadratic or inner product (str "quad" or "inner")
#Return: function taking a matrix A and returning the gradient of the objective corresponding to quadratic or inner product cost 
def getGrad(x,y,wtx,wty,reg,cost,sinkhornOpts):
    if cost == "quad":
        constantCost = computeConstantCost(x,y)
        grad = lambda A : gradientQuadratic(x,y,wtx,wty,A,constantCost,reg,sinkhornOpts)
    elif cost == "inner":
        grad = lambda A : gradientInnerProduct(x,y,wtx,wty,A,reg,sinkhornOpts)
    else: 
        raise ValueError("Supported costs are quadratic and inner product")
    return grad 

#Compute the EGW distance and plan when the variational objective is convex
#Input:     x: support points of mu (N_x x d_x)
        raise ValueError("Valid costs are 'quad' and 'inner'")
