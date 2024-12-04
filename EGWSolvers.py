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
#           y: support points of nu (N_y x d_y)
#         wtx: probability mass attributed to the N_x points of mu (N_x x 1)
#         wty: probability mass attributed to the N_y points of nu (N_y x 1) 
#         reg: regularization parameter for entropic Gromov-Wasserstein (float > 0)
#        cost: quadratic or inner product (str "quad" or "inner")
#           A: initial point (d_x x d_y)
#           L: Lipschitz constant of gradient of objective (float > 0) see Theorem 6/Corollary 34 in the paper
#      center: whether or not to center the distributions before running gradient alg. (boolean)
#       delta: tolerance for termination condition, norm of gradient < delta (float > 0)
#sinkhornOpts: Options to pass to the POT implementation of Sinkhorn's algorithm (see https://pythonot.github.io/_modules/ot/bregman/_sinkhorn.html#sinkhorn)
#Return: EGW distance and corresponding plan 
def EGWConvex(x,y,wtx,wty,reg,cost,A = None,L = None,center = True,delta = 1e-6,sinkhornOpts = {}):
    (Nx,dx) = np.shape(x)
    (Ny,dy) = np.shape(y)
    if A is None:
        A = np.zeros((dx,dy))
    if center == True:
        x = x - mean(x,wtx)
        y = y - mean(y,wty)
    if L == None and cost == "quad":
        L = 64
    elif L == None and cost == "inner":
        L = 16
    else:
        raise ValueError("costs are 'quad' or 'inner'")
    grad = getGrad(x,y,wtx,wty,reg,cost,sinkhornOpts)
    
    M = (np.dot(np.sum(x**2,axis=1),wtx)*np.dot(np.sum(y**2,axis=1),wty))**(0.5)+1e-5

    Ak = A
    lk = 0
    Gk = 1
    Wk = 0
    k = 0

    while np.linalg.norm(Gk)>delta:
        Gk,plan = grad(Ak)

        Wk = Wk+(k+1)/2*Gk

        px1 = Ak-Gk/L
        nm1 = np.linalg.norm(px1)
        Yk = min(1,M/(2*nm1))*px1

        nm2 = np.linalg.norm(Wk)/L
        Zk = -1/L*min(1,M/(2*nm2))*Wk

        Ak = 2/(k+3)*Zk+(k+1)/(k+3)*Yk

        k = k+1
        
    _,plan = grad(Yk)

    if cost == "quad":
        A_1,A_2=factorizedSquareEuclidean(x)
        B_1,B_2=factorizedSquareEuclidean(y)
        c2 = -2*np.trace(np.dot(np.dot(np.dot(B_2, plan.T), A_1), np.dot(np.dot(A_2, plan), B_1)))
        c1 = np.dot(np.dot(wtx,np.dot(A_1,A_2)**2),wtx)+np.dot(wty,np.dot(np.dot(B_1,B_2)**2,wty))
        return  c1+c2+reg*(np.sum(plan*np.log(plan))-np.sum(wtx*np.log(wtx))-np.sum(wty*np.log(wty))),plan
    elif cost == "inner":    
        A_1,A_2=x,x.T
        B_1,B_2=y,y.T
    
        c2 = -2* np.trace(np.dot(np.dot(np.dot(B_2, plan.T), A_1), np.dot(np.dot(A_2, plan), B_1)))
        c1 = np.dot(np.dot(wtx,np.dot(A_1,A_2)**2),wtx)+np.dot(wty,np.dot(np.dot(B_1,B_2)**2,wty))
        return  c1+c2+reg*(np.sum(plan*np.log(plan))-np.sum(wtx*np.log(wtx))-np.sum(wty*np.log(wty))), plan

#Estimate the EGW distance and plan when the variational objective is not convex
#Input:     x: support points of mu (N_x x d_x)
#           y: support points of nu (N_y x d_y)
#         wtx: probability mass attributed to the N_x points of mu (N_x x 1)
#         wty: probability mass attributed to the N_y points of nu (N_y x 1) 
#         reg: regularization parameter for entropic Gromov-Wasserstein (float > 0)
#        cost: quadratic or inner product (str "quad" or "inner")
#           A: initial point (d_x x d_y)
#           L: Lipschitz constant of gradient of objective (float > 0) see Theorem 6/Corollary 34 in the paper
#      center: whether or not to center the distributions before running gradient alg. (boolean)
#       delta: tolerance for termination condition, norm of gradient < delta (float > 0)
#sinkhornOpts: Options to pass to the POT implementation of Sinkhorn's algorithm (see https://pythonot.github.io/_modules/ot/bregman/_sinkhorn.html#sinkhorn)
#Return: estimate of the EGW distance and corresponding plan (note that in the nonconvex regime, spurious minima exist so exact resolution is not guaranteed)
def EGWAdaptive(x,y,wtx,wty,reg,cost,A = None,L = None,center = True,delta = 1e-6,sinkhornOpts = {}):
    (Nx,dx) = np.shape(x)
    (Ny,dy) = np.shape(y)
    if A is None:
        A = np.zeros((dx,dy))
    if center == True:
        x = x - mean(x,wtx)
        y = y - mean(y,wty)
    if L == None and cost == "quad":
        L = max(64,32**2/reg*(fourthMoment(x,wtx)*fourthMoment(y,wty))**(0.5)-64)
    elif L == None and cost == "inner":
        L = max(16,64/reg*(fourthMoment(x,wtx)*fourthMoment(y,wty))**(0.5)-16)
    elif L == None:
        raise ValueError("costs are 'quad' or 'inner'")    

    grad = getGrad(x,y,wtx,wty,reg,cost,sinkhornOpts)
    
    M = (np.dot(np.sum(x**2,axis=1),wtx)*np.dot(np.sum(y**2,axis=1),wty))**(0.5)+1e-5

    Ak = A
    lk = 0
    Gk = 1
    Zk = 0
    k = 0

    while np.linalg.norm(Gk)>delta:
        Gk,plan=grad(Ak)

        px1=Ak-Gk/(2*L)
        nm1=np.linalg.norm(px1)
        Yk=min(1,M/(2*nm1))*px1

        px2=Zk-(k+1)*Gk/(4*L)
        nm2=np.linalg.norm(px2)
        Zk=min(1,M/(2*nm2))*px2

        Ak=2/(k+3)*Zk+(k+1)/(k+3)*Yk

        k=k+1

    _,plan = grad(Yk)
    if cost == "quad":
        _,plan = grad(Yk)
    
        A_1,A_2=factorizedSquareEuclidean(x)
        B_1,B_2=factorizedSquareEuclidean(y)
        c2 = -2*np.trace(np.dot(np.dot(np.dot(B_2, plan.T), A_1), np.dot(np.dot(A_2, plan), B_1)))
        c1 = np.dot(np.dot(wtx,np.dot(A_1,A_2)**2),wtx)+np.dot(wty,np.dot(np.dot(B_1,B_2)**2,wty))
        return  c1+c2+reg*(np.sum(plan*np.log(plan))-np.sum(wtx*np.log(wtx))-np.sum(wty*np.log(wty))),plan
    elif cost == "inner":    
        A_1,A_2=x,x.T
        B_1,B_2=y,y.T
    
        c2 = -2* np.trace(np.dot(np.dot(np.dot(B_2, plan.T), A_1), np.dot(np.dot(A_2, plan), B_1)))
        c1 = np.dot(np.dot(wtx,np.dot(A_1,A_2)**2),wtx)+np.dot(wty,np.dot(np.dot(B_1,B_2)**2,wty))
        return  c1+c2+reg*(np.sum(plan*np.log(plan))-np.sum(wtx*np.log(wtx))-np.sum(wty*np.log(wty))), plan
        
#Helper function which checks if the objective is convex according to the Theorem 6/Corollary 34 in the paper
#Input:     x: support points of mu (N_x x d_x)
#           y: support points of nu (N_y x d_y)
#         wtx: probability mass attributed to the N_x points of mu (N_x x 1)
#         wty: probability mass attributed to the N_y points of nu (N_y x 1) 
#         reg: regularization parameter for entropic Gromov-Wasserstein (float > 0)
#        cost: quadratic or inner product (str "quad" or "inner")
#           A: initial point (d_x x d_y)
#           L: Lipschitz constant of gradient of objective (float > 0) see Theorem 6/Corollary 34 in the paper
#      center: whether or not to center the distributions before running gradient alg. (boolean)
#       delta: tolerance for termination condition, norm of gradient < delta (float > 0)
#sinkhornOpts: Options to pass to the POT implementation of Sinkhorn's algorithm (see https://pythonot.github.io/_modules/ot/bregman/_sinkhorn.html#sinkhorn)
#Return: estimate of the EGW distance and corresponding plan (note that in the nonconvex regime, spurious minima exist so exact resolution is not guaranteed)
def EGWAuto(x,y,wtx,wty,reg,cost,A = None,L = None,center = True,delta = 1e-6,sinkhornOpts = {}):
    if center == True:
        x = x - mean(x,wtx)
        y = y - mean(y,wty)
    if cost == "quad":
        if np.sqrt(fourthMoment(x,wtx)*fourthMoment(y,wty))<reg/16:
            return EGWConvex(x,y,wtx,wty,reg,cost,A,L,False,delta,sinkhornOpts)
        else:
            return EGWAdaptive(x,y,wtx,wty,reg,cost,A,L,False,delta,sinkhornOpts)
    elif cost == "inner":
        if np.sqrt(fourthMoment(x,wtx)*fourthMoment(y,wty))<reg/4:
            return EGWConvex(x,y,wtx,wty,reg,cost,A,L,False,delta,sinkhornOpts)
        else:
            return EGWAdaptive(x,y,wtx,wty,reg,cost,A,L,False,delta,sinkhornOpts)
    else:
        raise ValueError("Valid costs are 'quad' and 'inner'")
