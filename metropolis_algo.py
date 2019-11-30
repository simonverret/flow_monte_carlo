#%%
import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy

class SpinConfiguration():
    def __init__(self, lenght):
        self.Lx = lenght
        self.Ly = lenght
        self.L = lenght
        self.N = lenght*lenght
        self.S_array = None

        # Periodic boundary lookup table 
        self.P1 = np.arange(1,lenght+1)
        self.P1[-1]=0       # ex: if L = 10,  P1 = [1,2,3,4,5,6,7,8,9,0] 
        self.M1 = np.arange(-1,lenght-1)
        self.M1[0]=lenght-1 # ex: if L = 10,  M1 = [9,0,1,2,3,4,5,6,7,8]
    
    def initialize(self):
        self.S_array = np.ones([self.Lx, self.Ly],dtype=int)
        self.S_array[np.random.random([self.Lx,self.Ly])<=0.5] = -1

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

    def plot(self):
        plt.matshow(self.S_array, cmap='gray')
        M = round(self.compute_magnetization(),4)
        E = round(self.compute_energy(),4)
        plt.title('E='+str(E)+', M='+str(M))
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def naive_MC_update(self, temperature):
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

# if __name__=='__main__':

parser = argparse.ArgumentParser(description='Parameters for Monte-Carlo simulation:')
parser.add_argument("-L", dest ="L", type=int, default=16, help='Side of the LxL lattice')
parser.add_argument("-Tmin", dest ="Tmin", type=int, default=0.2)
parser.add_argument("-Tmax", dest ="Tmax", type=int, default=4)
parser.add_argument("-Tnum", dest ="Tnum", type=int, default=20)
parser.add_argument("-steps", dest ="steps", type=int, default=int(2e5))
parser.add_argument("-skip", dest ="skip_frac", type=float, default=0.2)
args = parser.parse_known_args()[0]


## MONTE-CARLO SIMULATION

S1 = SpinConfiguration(args.L)
S1.initialize()
# S1.plot()
print('Monte Carlo Simulation')
print(type(args))

T_list = np.linspace(args.Tmin,args.Tmax,args.Tnum)
M_list = np.zeros([args.steps, args.Tnum])
for tt, T in enumerate(T_list):
    print('  T=',T)
    S1.initialize()
    for step in range(args.steps):
        if step%(2e3)==0: print('    step',step,'of',args.steps)
        # if step%1000==0: S1.plot()
        S1.naive_MC_step(T)
        M_list[step,tt] = S1.compute_magnetization()

#%%

M_avg = np.zeros([args.steps, args.Tnum])
for tt, T in enumerate(T_list):
    for step in range(args.steps):
        M_avg[step,tt] = np.average(M_list[:step,tt])

for tt,T in enumerate(T_list):
    plt.title('T='+str(round(T,3)))
    plt.plot(M_list[:,tt], alpha=0.5)
    plt.plot(M_avg[:,tt], linewidth=2)
    plt.ylim(-1,1)
    plt.show()

#%% COMPARE TO EXACT SOLUTION

def exact_magnetization(T):
    T_c = 2.269185 # *J/k
    out = np.zeros(T.shape)
    out[T<T_c] = (1.0 - np.sinh(2.0/T[T<T_c])**(-4))**(1.0/8)
    return out

skip = int(args.skip_frac*args.steps)
absM = np.average(np.abs(M_avg[skip:,:]),axis=0)
errM = np.var(M_avg[skip:,:],axis=0)

plt.errorbar(T_list, absM, yerr = errM, linewidth=1, marker='o', markersize=4, elinewidth=1, capsize=2)
plt.axvline(x=2.0/np.log(1.0+np.sqrt(2.0)), linewidth=1, color='gray', linestyle='--')
plt.plot(T_list, exact_magnetization(T_list))
plt.xlabel(r'Temperature ($k_{\rm B}T/J$)')
plt.ylabel('Magnetization per spin')
plt.title(f"L = {args.L}")
plt.show()


#%% BINNING ANALYSIS TO CORRECT ERROR BARS

num_levels = np.int(np.log2( (args.steps - skip)/4 )) + 1
print(num_levels)
errM_binned = np.zeros([args.Tnum, num_levels])
binned = M_list[skip:,:]
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
T_dense = np.linspace(0.01,4,1000)
plt.plot(T_dense, exact_magnetization(T_dense))
plt.xlabel(r'Temperature ($k_{\rm B}T/J$)')
plt.ylabel('Magnetization per spin')
plt.title(f"L = {args.L}")
plt.show()

