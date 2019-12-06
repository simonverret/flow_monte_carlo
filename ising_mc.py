#%%
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy


#%%
def exact_ising_magnetization(T):
    ''' T : np array of temperature ''' 
    T_c = 2.269185 # *J/k
    out = np.zeros(T.shape)
    out[T<T_c] = (1.0 - np.sinh(2.0/T[T<T_c])**(-4))**(1.0/8)
    return out

class SpinConfiguration():
    def __init__(self, lenght):
        ''' This class will manages the updates, energies and magnetization computation
            lenght : side lenght of the square lattice  '''
        self.Lx = lenght
        self.Ly = lenght
        self.L = lenght
        self.N = lenght*lenght
        self.S_array = None
        self.generate_neighbors_dict() # defines self.neighbors

        # Periodic boundary lookup table 
        self.P1 = np.arange(1,lenght+1)
        self.P1[-1]=0       # ex: if L = 10,  P1 = [1,2,3,4,5,6,7,8,9,0] 
        self.M1 = np.arange(-1,lenght-1)
        self.M1[0]=lenght-1 # ex: if L = 10,  M1 = [9,0,1,2,3,4,5,6,7,8]
    
    def generate_neighbors_dict(self):
        self.neighbors = {}
        for i in range(self.L):
            for j in range(self.L):
               self.neighbors[i,j] = [
                        (self.P1[i],j),
                        (self.M1[i],j),
                        (i,self.P1[j]),
                        (i,self.M1[j])
                    ]
        return self.neighbors

    def initialize(self):
        self.S_array = np.ones([self.Lx, self.Ly],dtype=int)
        self.S_array[np.random.random([self.Lx,self.Ly])<=0.5] = -1

    def naive_MC_update(self, temperature):
        ''' perform flip of one random spin according the the Metropolis Hasting
            temperature : scalar (not broadcastable)'''
        x = np.random.randint(0,self.Lx)
        y = np.random.randint(0,self.Ly)
        delta_E  = 2 * self.S_array[x,y] * self.S_array[self.M1[x],y]
        delta_E += 2 * self.S_array[x,y] * self.S_array[self.P1[x],y]
        delta_E += 2 * self.S_array[x,y] * self.S_array[x,self.M1[y]]
        delta_E += 2 * self.S_array[x,y] * self.S_array[x,self.P1[y]]
        #metropolis test
        if np.random.random() < np.exp(-delta_E/temperature):
            self.S_array[x,y] *= -1

    def naive_MC_step(self, temperature):
        for update in range(self.N):
            self.naive_MC_update(temperature)
        return self.S_array

    def wolff_step(self, temperature, print_mask=False):
        prob_add = 1-np.exp(-2/temperature)

        i0 = np.random.randint(0,self.L)
        j0 = np.random.randint(0,self.L)
        spin0 = self.S_array[i0,j0]
        
        cluster_mask = np.zeros((self.L,self.L), dtype=bool)
        cluster_mask[i0,j0] = 1

        pocket = [(i0,j0)]
        while pocket != []:
            new_pocket = []
            for i,j in pocket: 
                for n,m in self.neighbors[i,j]:
                    aligned = self.S_array[n,m] == spin0
                    out_pkt = (n,m) not in pocket
                    out_cls = not cluster_mask[n,m]
                    picked  = np.random.random() < prob_add
                    if picked and aligned and out_pkt and out_cls:
                        new_pocket.append( (n,m) ) 
                        cluster_mask[n,m] = True
            pocket = new_pocket
        
        self.S_array[cluster_mask] *= -1

    def compute_energy(self):
        energy = 0.
        for x in range(self.Lx):
            for y in range(self.Ly):
                energy -= .5*self.S_array[x, y] * self.S_array[self.M1[x], y] /self.N
                energy -= .5*self.S_array[x, y] * self.S_array[self.P1[x], y] /self.N
                energy -= .5*self.S_array[x, y] * self.S_array[x, self.M1[y]] /self.N
                energy -= .5*self.S_array[x, y] * self.S_array[x, self.P1[y]] /self.N
        return energy

    def compute_magnetization(self):
        return np.sum(self.S_array)/self.N    

    def print(self):
        fig = plt.figure(figsize=[2,2])
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.matshow(self.S_array, cmap='gray')
        
        M = round(self.compute_magnetization(),4)
        E = round(self.compute_energy(),4)
        fig.suptitle('E='+str(E)+', M='+str(M), fontsize=10)
        return fig


def monte_carlo_simulation(L, Tmin, Tmax, Tnum, steps, wolff=True):
    S1 = SpinConfiguration(L)
    S1.initialize()

    print('Monte Carlo Simulation')

    T_list = np.linspace(Tmin,Tmax,Tnum)
    M_list = np.zeros([steps, Tnum])
    for tt, T in enumerate(T_list):
        print('  T=',round(T,2))
        S1.initialize()
        for step in range(steps):
            if (step+1)%(steps//10)==0 and (steps * L > 2e4 ):
                print('    step',step+1,'of',steps)
            if wolff : S1.wolff_step(T)
            else:      S1.naive_MC_step(T)
            M_list[step,tt] = S1.compute_magnetization()
    return M_list



# if __name__=='__main__':

parser = argparse.ArgumentParser(description='Parameters for Monte-Carlo simulation:')
parser.add_argument("-L", dest ="L", type=int, default=64, help='Side of the LxL lattice')
parser.add_argument("-Tmin", dest ="Tmin", type=int, default=1)
parser.add_argument("-Tmax", dest ="Tmax", type=int, default=3.6)
parser.add_argument("-Tnum", dest ="Tnum", type=int, default=14)
parser.add_argument("-steps", dest ="steps", type=int, default=int(5e2))
parser.add_argument("-skip", dest ="skip_frac", type=float, default=0.4)
args = parser.parse_known_args()[0]
name = f'L{args.L}_T{args.Tmin}-{args.Tmax}-{args.Tnum}_step{args.steps}'

# MONTE-CARLO SIMULATION
M_list = monte_carlo_simulation(args.L, args.Tmin, args.Tmax, args.Tnum, args.steps)

# SAVE RESULTS
try: os.stat('results')
except: os.mkdir('results')  
path = os.path.join('results',name+'.csv')
np.savetxt(path, M_list, delimiter='\t')


T_list = np.linspace(args.Tmin,args.Tmax,args.Tnum)
M_avg = np.zeros([args.steps, args.Tnum])
M_abs_avg = np.zeros([args.steps, args.Tnum])
M_rms_avg = np.zeros([args.steps, args.Tnum])

def running_mean(array_in):
    sum = np.cumsum(array_in,axis=1)
    running_denom = np.arange(array_in.shape[1])+1
    return 

for tt, T in enumerate(T_list):
    for step in range(args.steps):
        M_avg[step,tt] = np.average(M_list[:step,tt])
        M_abs_avg[step,tt] = np.average(np.abs(M_list[:step,tt]))
        M_rms_avg[step,tt] = np.sqrt(np.average(M_list[:step,tt]**2))

for tt,T in enumerate(T_list):
    if T in [1.4,2,2.2,2.4,2.6,3.2]:
        plt.title('T='+str(round(T,3)))
        plt.plot(M_list[:,tt], alpha=0.4)
        plt.plot(M_rms_avg[:,tt], linewidth=2, label='rms')
        plt.plot(M_abs_avg[:,tt], linewidth=2, label='abs')
        plt.plot(M_avg[:,tt], linewidth=2, label='avg')
        plt.legend(ncol=1, loc=(0.78,0.02))
        plt.ylim(-1,1)
        plt.show()

#%% COMPARE TO EXACT SOLUTION
skip = int(args.skip_frac*args.steps)
avgM = np.abs(np.average(M_list[skip:,:],axis=0))
absM = np.average(np.abs(M_list[skip:,:]),axis=0)
rmsM = np.average(np.sqrt(M_list[skip:,:]**2),axis=0)

err_avgM = np.var(M_list[skip:,:],axis=0)
err_absM = np.var(np.abs(M_list[skip:,:]),axis=0)
err_rmsM = np.var(np.sqrt(M_list[skip:,:]**2),axis=0)

T_dense = np.linspace(args.Tmin,args.Tmax,1000)
plt.plot(T_dense, exact_ising_magnetization(T_dense))
plt.errorbar(T_list, avgM, yerr = err_avgM, label='avg', linewidth=1, marker='o', markersize=4, elinewidth=1, capsize=2)
plt.errorbar(T_list, absM, yerr = err_absM, label='abs', linewidth=1, marker='o', markersize=4, elinewidth=1, capsize=2)
plt.errorbar(T_list, rmsM, yerr = err_rmsM, label='rms', linewidth=1, marker='o', markersize=4, elinewidth=1, capsize=2)
plt.axvline(x=2.0/np.log(1.0+np.sqrt(2.0)), linewidth=1, color='gray', linestyle='--')
plt.xlabel(r'Temperature ($k_{\rm B}T/J$)')
plt.legend(ncol=1, loc=(0.78,0.02))

plt.ylabel('Magnetization per spin')
plt.title(f"L = {args.L}")
plt.show()

#%%
plt.ylabel('Magnetization per spin')
T_dense = np.linspace(args.Tmin,args.Tmax,1000)
plt.plot(T_dense, exact_ising_magnetization(T_dense))
plt.axvline(x=2.0/np.log(1.0+np.sqrt(2.0)), linewidth=1, color='gray', linestyle='--')
plt.xlabel(r'Temperature ($T$)')
plt.title('T_c = 2.2691...')

#%%

for i in range(len(T_list)):
    plt.hist(M_list[:,i],bins=30)
    plt.show()
#%%
# BINNING ANALYSIS TO CORRECT ERROR BARS

binned = M_list[skip:,:]
num_levels = np.int(np.log2( (binned.shape[0])/4 )) + 1
errM_binned = np.zeros([args.Tnum, num_levels])
for n in range(num_levels):
    skip1 = binned.shape[0]%2
    errM_binned[:,n] = np.std(binned,axis=0)/np.sqrt(binned.shape[0])
    binned = ( binned[skip1::2,:] + binned[skip1+1::2,:] )/2

for tt,T in enumerate(T_list[6:15]):
    plt.plot(np.abs(errM_binned[6+tt,:]),'-o', linewidth=0.5, markersize=4, label='$k_{\mathrm{B}}T/J = %4.2f$'%T)
plt.xlabel('bin level')
plt.ylabel('error');
plt.legend(ncol=1, loc=(0.02,0.18))
plt.show()


plt.errorbar(T_list, absM, yerr = np.max(errM_binned,axis=1), linewidth=1, marker='o', markersize=4, elinewidth=1, capsize=2)
plt.axvline(x=2.0/np.log(1.0+np.sqrt(2.0)), linewidth=1, color='gray', linestyle='--')
T_dense = np.linspace(args.Tmin,args.Tmax,1000)
plt.plot(T_dense, exact_ising_magnetization(T_dense))
plt.xlabel(r'Temperature ($k_{\rm B}T/J$)')
plt.ylabel('Magnetization per spin')
plt.title(f"L = {args.L}")


#%% SAVE COMPUTATION
name = f'L{args.L}_T{args.Tmin}-{args.Tmax}-{args.Tnum}_step{args.steps}_skip{args.skip_frac}'
plt.savefig(name+'.pdf')

