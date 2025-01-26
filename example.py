from EGWSolvers import *
 
Nx,dx = 1000,2
Ny,dy = 1000,6
reg = 1
rng = np.random.default_rng(12345)
 
x = rng.normal(0,.1,(Nx,dx))
y = rng.normal(0,.11,(Ny,dy))
wtx = rng.random(Nx)
wtx/=np.sum(wtx)
wty = rng.random(Ny)
wty/=np.sum(wty)

x = x - mean(x,wtx)
y = y - mean(y,wty)

A = np.zeros((dx,dy))
L = 64
center = True
delta = 5e-8
sinkhornOpts = {"method":"sinkhorn", "numItermax":10000}
cost,plan = EGWAuto(x,y,wtx,wty,reg,"quad",A = A,L = L,center = True,delta = delta,sinkhornOpts = sinkhornOpts)
