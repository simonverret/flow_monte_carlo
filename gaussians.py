#%%
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad




SQRT2PI = np.sqrt(2*np.pi)
def gaussian(x, mu=0, sigma=1, amp = 1):
    return amp*torch.exp(-(x-mu)**2/(2*sigma**2))/(SQRT2PI*sigma)

def uniform(x, a=0, b=1, amp = 1):
    out = torch.zeros_like(x)
    x_np = x.detach().numpy()
    mask = ( x_np > a and x_np < b )
    out[mask] = amp/(b-a) 
    return out

#%% MODEL DEFINITION

LO = 0.1
HI = 0.9

def logit(x):
        return torch.log(x/(1-x))

class OneToOne(nn.Module):
    def __init__(self, h=1, bias=True):
        super().__init__()

        # weights will have to be normalized (see below)
        self.pre_a = torch.Tensor(h,1)          # allocate
        self.pre_a = self.pre_a.normal_(0,1)    # initialize
        self.pre_a = nn.Parameter(self.pre_a)   # make optimizable

        self.pre_w = torch.Tensor(1,h)          # allocate
        self.pre_w = self.pre_w.normal_(0,1)    # initialize
        self.pre_w = nn.Parameter(self.pre_w)   # make optimizable

        # bias
        self.bias = bias
        self.b = torch.zeros(h)                 # allocate
        self.b = nn.Parameter(self.b)       # make optimizable


    def forward(self, x):
        # normalize weights (make both positive and normalize w)
        a = F.softplus(self.pre_a)
        w = F.softmax(self.pre_w, dim=-1)

        # apply
        if self.bias: out = F.linear(x,a,self.b)        
        else: out = F.linear(x,a,None)
        out = torch.sigmoid(out)
        out = F.linear(out,w,None)
        out = logit(out)
        return out


class OneToOneStack(nn.Module):
    def __init__(self, h_list=[1,1], bias_list=[True,True]):
        super().__init__()
        self.num_blocks = len(h_list)

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(
                OneToOne(h_list[i],bias_list[i])
            )
        
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out

def loss_fn(df_dx, ptarget):
    ''' DKL( p_model(y) || p_target(y) ) = 
        E[ log p_0(x) - log df(x)/dx - log p(f(x)) ]
        thus it is sufficient to minimize 
        -E[ log df(x)/dx + log p(f(x)) ] 
        
        so 
        df_dx : the derivative of the model at x 
        ptarget is the probability under the target distribution
        of the y computed by the model '''
    bs = df_dx.shape[0]
    log_df_dx = torch.log(df_dx)
    log_ptarget = torch.log(ptarget)
    avg = torch.sum( log_df_dx + log_ptarget, dim=0) /bs
    return -avg

# training
#
## experiment parameters ####
def target_distribution(x_in):
    return gaussian(x_in,mu=-5,amp=0.1)+ gaussian(x_in,mu=-3,amp=0.2) + gaussian(x_in,mu=3,amp=0.5)+ gaussian(x_in,mu=6,amp=0.2)

batch_size = 10000
num_epoch = 10000
# model = OneToOne(h=3, bias=False)
model = OneToOneStack(h_list=[3,1], bias_list=[True,False])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
## experiment parameter end ####


ones = torch.ones((batch_size,1))#.to(device)

for epoch in range(num_epoch):
    
    # sample the latent distribution (gaussian)
    inputs = torch.Tensor(batch_size,1).normal_(0,4).requires_grad_()
    
    optimizer.zero_grad()
    
    y_pred = model(inputs)
    p_target = target_distribution(y_pred)
    df_dx = grad(y_pred, inputs, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    loss = loss_fn(df_dx, p_target)

    loss.backward()
    optimizer.step()

    if epoch%(num_epoch//20)==0: 
        print(f'epoch {epoch} loss={loss}')
        
#%%

NGRID = 200
NSAMP = 100000

# data latent space
x_grid = torch.linspace(-5,5,NGRID)
p_latent = gaussian(x_grid, 0, 4)
x_samples = torch.Tensor(NSAMP,1).normal_(0,4)

# data model
m_out = model(x_grid.view(NGRID,1))

# data target
y_grid = torch.linspace(-10,10,NGRID)
y_samples = model(x_samples)
f_target = target_distribution(y_grid)

# detach
x_grid = x_grid.detach().numpy()
p_latent = p_latent.detach().numpy()
x_samples = x_samples.detach().numpy()

m_out = m_out.detach().numpy()

y_grid = y_grid.detach().numpy()
y_samples = y_samples.detach().numpy()
f_target = f_target.detach().numpy()

# plot
fig = plt.figure(figsize=(6, 6)) 
gs = gridspec.GridSpec(3, 3) 

ax_x = plt.subplot(gs[2,:2])
ax_y = plt.subplot(gs[:2,2])
ax_m = plt.subplot(gs[:2,0:2])
ax_m.set_title('y = model(x)')
plt.setp(ax_m.get_xticklabels(), visible=False)
plt.setp(ax_y.get_yticklabels(), visible=False)
ax_x.set_xlabel('x')
ax_m.set_ylabel('y')
ax_m.set_ylim([-10,10])

ax_x.hist(x_samples, bins=50, density=1)
ax_x.plot(x_grid,p_latent)
ax_m.plot(x_grid, m_out)
ax_y.hist(y_samples, bins=50, density=1,orientation='horizontal')
ax_y.plot(f_target,y_grid,label='target')

plt.show()
# plt.savefig('sucess_plot_02.pdf')


## learn monotonic NN for two gaussians from one gaussian random input

## plot the learned function

## do monte carlo 











#%%
N = 10000
data1 = np.random.normal(size=N)

split2 = 2
d1 = np.random.normal(size=N//2)+split2
d2 = np.random.normal(size=N//2)-split2
data2 = np.concatenate([d1,d2])

split3 = 6
d1 = np.random.normal(size=N//2)+split3
d2 = np.random.normal(size=N//2)-split3
data3 = np.concatenate([d1,d2])

x = torch.linspace(-10,10,1000)
gauss1 = gaussian(x).detach().numpy()
gauss2 = gaussian(x,mu=-split2,amp=0.5)+gaussian(x,mu=split2,amp=0.5)
gauss2 = gauss2.detach().numpy()
gauss3 = gaussian(x,mu=-split3,amp=0.5)+gaussian(x,mu=split3,amp=0.5)
gauss3 = gauss3.detach().numpy()
x = x.detach().numpy()


fig = plt.figure(figsize=(6,2))
fig.suptitle(f'{N} samples')

ax1 = fig.add_subplot(1,3,1)
ax1.plot(x, gauss1)
ax1.hist(data1, bins=100, density=1)
ax1.set_xlabel('x')
ax1.set_ylabel('p(x)')

ax2 = fig.add_subplot(1,3,2, sharex=ax1, sharey=ax1)
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.plot(x, gauss2)
ax2.hist(data2, bins=100, density=1)
ax2.set_xlabel('x')

ax3 = fig.add_subplot(1,3,3, sharex=ax1, sharey=ax1)
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.plot(x, gauss3)
ax3.hist(data3, bins=100, density=1)
ax3.set_xlabel('x')

plt.savefig('sucess_plot_02.pdf')



