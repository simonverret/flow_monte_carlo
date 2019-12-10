#%% IMPORT MODULES

import numpy as np
from numpy import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc


# UTIL

SQRT2PI = np.sqrt(2*np.pi)

def gaussian(x, mu=0, sigma=1, amp = 1):
    return amp*torch.exp(-(x-mu)**2/(2*sigma**2))/(SQRT2PI*sigma)

def uniform(x, a=0, b=1, amp = 1):
    out = torch.ones_like(x)
    out[(x < a).to(torch.uint8)] = 0
    out[(x > b).to(torch.uint8)] = 0
    return out

def logit(x):
        return torch.log(x/(1-x))


# MODEL

class OneToOne(nn.Module):
    def __init__(self, h=1, bias=True):
        super().__init__()
        # weights have to be normalized at every forward pass
        pre_a = torch.empty(h,1)            # allocate
        torch.nn.init.normal_(pre_a,0,1)          # initialize
        self.pre_a = nn.Parameter(pre_a)    # make optimizable
        
        pre_w = torch.empty(1,h)            # allocate
        torch.nn.init.normal_(pre_w,0,1)          # initialize
        self.pre_w = nn.Parameter(pre_w)    # make optimizable
        
        if bias:
            b = torch.zeros(h)              # allocate
            self.b = nn.Parameter(b)        # make optimizable
        else:
            self.b = None

    def forward(self, x):
        # normalize weights (make both positive and normalize w)
        a = F.softplus(self.pre_a)
        w = F.softmax(self.pre_w, dim=-1)

        out = F.linear(x,a,self.b)        
        out = torch.sigmoid(out)
        out = F.linear(out,w,None)
        out = logit(out)
        return out

class SimplerOneToOne(nn.Module):
    def __init__(self, h=1, bias=True):
        super().__init__()
        # weights have to be positive at every forward pass
        pre_a = torch.empty(h,1)            # allocate
        torch.nn.init.uniform_(pre_a,0.9,1.1)        # initialize
        self.pre_a = nn.Parameter(pre_a)    # make optimizable
        
        pre_w = torch.empty(1,h)            # allocate
        torch.nn.init.uniform_(pre_w,0.9,1.1)        # initialize
        self.pre_w = nn.Parameter(pre_w)    # make optimizable
        
        b = torch.empty(h)               # allocate
        torch.nn.init.uniform_(b,-2,2)
        self.b = nn.Parameter(b)            # make optimizable
        
        r = torch.ones(1)                   # allocate
        self.r = nn.Parameter(r)            # make optimizable

    def forward(self, x):
        # normalize weights (make both positive and normalize w)
        a = F.softplus(self.pre_a)
        w = F.softplus(self.pre_w)
        r = F.sigmoid(self.r)

        out = F.linear(x,a,self.b)        
        out = torch.sigmoid(out)
        out = F.linear(out,w,None)
        out = self.r * x + out
        return out

class OneToOneStack(nn.Module):
    def __init__(self, h_list=[1,1], bias_list=[True,True]):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(h_list)):
            h = h_list[i]
            b = bias_list[i]
            self.blocks.append(OneToOne(h,b))
        
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


# LOSS

def loss_fn(df_dx, ptarget):
    ''' DKL( p_model(y) || p_target(y) ) = 
        E[ log p_0(x) - log |df(x)/dx| - log p(f(x)) ]
        thus it is sufficient to minimize 
        -E[ log df(x)/dx + log p(f(x)) ] '''
    bs = df_dx.shape[0]
    log_df_dx = torch.log(df_dx + 1e-9)
    log_ptarget = torch.log(ptarget + 1e-9)
    avg = torch.sum( (log_df_dx + log_ptarget)/bs, dim=0)
    return -avg


# ARGS ##############################################
def target_distribution(x_in):
    out  = gaussian(x_in, mu=-5, sigma=0.8, amp=0.1)
    out += gaussian(x_in, mu=-3, sigma=1.2, amp=0.2) 
    out += gaussian(x_in, mu= 3, sigma=0.5, amp=0.5)
    out += gaussian(x_in, mu= 6, sigma=1.0, amp=0.2)
    return out

batch_size = 1000
num_epoch = 10000
model = OneToOneStack(h_list=[8], bias_list=[True,True,True])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

sigma0 = 3
# ARGS END #########################################

# TRAINING
ones = torch.ones((batch_size,1)) # dummy gradient for chain rule

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)
# ax.set_xticks([])
# ax.set_yticks([])

for epoch in range(num_epoch):
    # the data is random latent variable
    inputs = torch.Tensor(batch_size,1).normal_(0,sigma0).requires_grad_()
    
    optimizer.zero_grad()
    
    y_pred = model(inputs)
    p_target = target_distribution(y_pred)
    
    df_dx = grad(y_pred, inputs, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    loss = loss_fn(df_dx, p_target)
    loss.backward()
    optimizer.step()

    if epoch%(num_epoch//20)==0: 
        print(f'epoch {epoch}')
        print(f'  loss     : {round(loss.item(),3)}')
        print(f'  jacobian : min={round(df_dx.min().item(),3)}, max={round(df_dx.max().item(),3)}, avg={round((df_dx.sum()/batch_size).item(),3)}')
        # x_grid = torch.linspace(-10,10,100)
        # m_out = model(x_grid.view(100,1))
        # ax.clear()
        # ax.plot(x_grid.detach().numpy(),m_out.detach().numpy(), color='#ff0000')
        # plt.draw()
        # plt.pause(.001)


# PLOT DATA
NGRID = 1000
NSAMP = 10000

x_grid = torch.linspace(-10,10,NGRID)
p_latent = gaussian(x_grid,0,sigma0)
x_samples = torch.Tensor(NSAMP,1).normal_(0,sigma0)
m_out = model(x_grid.view(NGRID,1))
y_grid = torch.linspace(-10,10,NGRID)
y_samples = model(x_samples)
f_target = target_distribution(y_grid)

# torch detach
x_grid = x_grid.detach().numpy()
p_latent = p_latent.detach().numpy()
x_samples = x_samples.detach().numpy()
m_out = m_out.detach().numpy()
y_grid = y_grid.detach().numpy()
y_samples = y_samples.detach().numpy()
f_target = f_target.detach().numpy()


# PLOT
rc('text', usetex=True)
rc('axes', linewidth=0.5)
rc('xtick.major', width=0.5)
rc('ytick.major', width=0.5)

fig = plt.figure(figsize=(4.5, 4)) 
gs = gridspec.GridSpec(4, 4) 

ax_x = plt.subplot(gs[3,:3])
ax_y = plt.subplot(gs[:3,3])
ax_m = plt.subplot(gs[:3,:3], sharex=ax_x, sharey=ax_y)
ax_m.set_title(r'$y = f_{\theta}(x)$')
plt.setp(ax_m.get_xticklabels(), visible=False)
plt.setp(ax_y.get_yticklabels(), visible=False)
plt.setp(ax_y.get_xticklabels(), visible=False)
plt.setp(ax_x.get_yticklabels(), visible=False)
ax_x.set_yticks([])
ax_y.set_xticks([])
ax_x.set_xlabel('$x$')
ax_m.set_ylabel('$y$')
ax_y.set_ylim([-8,8])
ax_x.set_xlim([-10,10])
ax_x.set_xticks([-10,-5,0,5,10])

c_smpx = '#0088bb'
c_dstx = '#00aa00'
c_smpy = '#0088bb'
c_dsty = '#ff0000'
c_mdl = '#0000ff'
alph = 0.4

ax_x.hist(x_samples, bins=500,density=1, color=c_smpx,alpha=alph)
ax_x.plot(x_grid,p_latent, color=c_dstx)
ax_y.hist(y_samples, bins=500,density=1, color=c_smpy,alpha=alph, orientation='horizontal')
ax_m.plot(x_grid, m_out, color=c_mdl)

ones = torch.ones((NGRID,1))
x_grid = torch.linspace(-15,15,NGRID).view(NGRID,1)
x_grid.requires_grad_()
y_out = model(x_grid)
df_dx = grad(y_out, x_grid, grad_outputs=ones, create_graph=True)[0]
p_y_out = gaussian(x_grid,0,sigma0)/df_dx

ax_y.plot(p_y_out.detach().numpy(),y_out.detach().numpy(), linewidth=0.7, color=c_mdl, label='model')
ax_y.plot(f_target,y_grid, color=c_dsty, label='target')
ax_y.legend(loc=(0,-0.25), handlelength=0.8)


# plt.show()
plt.savefig('sucess_plot_05.pdf')


#%%
NGRID=400

ones = torch.ones((NGRID,1))
x_grid = torch.linspace(-15,15,NGRID).view(NGRID,1)
x_grid.requires_grad_()
y_out = model(x_grid)
df_dx = grad(y_out, x_grid, grad_outputs=ones, create_graph=True)[0]
p_y_out = gaussian(x_grid,0,sigma0)/df_dx

plt.plot(y_out.detach().numpy(), p_y_out.detach().numpy())
plt.hist(y_samples, bins=500,density=1, color=c_smpy,alpha=alph)
plt.show()

#%%
# MONTE-CARLO 

def flow_mcmc(num_update=1000):
    accept_ratio=0
    samples = np.zeros(num_update)

    x_a = torch.Tensor(1).normal_(0,sigma0).requires_grad_()
    a = model(x_a)
    df_dx_a = grad(a, x_a, grad_outputs=ones, create_graph=True)[0]
    p_b_to_a = gaussian(x_a,0,sigma0)/df_dx_a
    p_a = target_distribution(a)

    for i in range(num_update):
        x_b = torch.Tensor(1).normal_(0,sigma0).requires_grad_()
        b = model(x_b)
        df_dx_b = grad(b, x_b, grad_outputs=ones, create_graph=True)[0]
        p_a_to_b = gaussian(x_b,0,sigma0)/df_dx_b
        p_b = target_distribution(torch.Tensor([b]))
    
        if np.random.rand() < (p_b_to_a * p_b) /(p_a_to_b * p_a):
            a = b
            accept_ratio += 1/num_update
            p_b_to_a = p_a_to_b
            p_a = p_b

        samples[i] = a.item()

    print(f'acceptation ratio = {accept_ratio}')
    return samples, accept_ratio

def metropolis_hastings_mcmc(num_update=1000, var=1, prop='recenter'):
    a = 0
    accept_ratio = 0
    samples = np.zeros(num_update)
    for i in range(num_update):
        if prop == 'recenter':
            b = np.random.normal(a, var)
            p_a_to_b = 1
            p_b_to_a = 1
        elif prop == 'gaussian':
            b = np.random.normal(0, var)
            p_a_to_b = gaussian(torch.Tensor([b]),0,var)
            p_b_to_a = gaussian(torch.Tensor([a]),0,var)
        elif prop == 'uniform':
            b = np.random.uniform(-var,var)
            p_a_to_b = 1
            p_b_to_a = 1
       
        p_a = target_distribution(torch.Tensor([a]))
        p_b = target_distribution(torch.Tensor([b]))
        # metropolis test
        if np.random.rand() < (p_b_to_a * p_b) /(p_a_to_b * p_a):
            a = b
            accept_ratio += 1/num_update
        samples[i] = a
    print(f'acceptation ratio = {accept_ratio}')
    return samples, accept_ratio

mh1_samples, accept_r1 = metropolis_hastings_mcmc(1000, var=0.5, prop='recenter')
mh2_samples, accept_r2 = metropolis_hastings_mcmc(1000, var=sigma0, prop='gaussian')
mh3_samples, accept_r3 = flow_mcmc(1000)

#%%
fig = plt.figure(figsize=(4.5, 4)) 
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.set_ylim(0,0.5)
ax1.set_ylabel('$y$')
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.hist(mh1_samples, bins=500, density=1, color=c_smpx, alpha=alph)
ax1.plot(y_grid, f_target, color=c_dsty, label='target')

ax2.set_ylim(0,0.5)
ax2.set_ylabel('$y$')
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.hist(mh2_samples, bins=500, density=1, color=c_smpx, alpha=alph)
ax2.plot(y_grid, f_target, color=c_dsty, label='target')

ax3.set_ylim(0,0.5)
ax3.set_ylabel('$y$')
ax3.set_xlabel('$x$')
ax3.hist(mh3_samples, bins=500, density=1, color=c_smpx, alpha=alph)
ax3.plot(y_grid, f_target, color=c_dsty, label='target')

# plt.show()
plt.savefig('mc_plot_01.pdf')