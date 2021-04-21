import math
import numpy as np
import argparse
import os
import parser as read

def enable_cache():
    if not os.path.exists('perm'):
        os.makedirs('perm')
    if not os.path.exists('scratch'):
        os.makedirs('scratch')


def nuclear_repulsion(enuc):
    with open(enuc, 'r') as f:
       e = float(f.read())
    return e

def eint(a,b,c,d):
    if a > b: ab = a*(a+1)/2 + b
    else: ab = b*(b+1)/2 + a
    if c > d: cd = c*(c+1)/2 + d
    else: cd = d*(d+1)/2 + c
    if ab > cd: abcd = ab*(ab+1)/2 + cd
    else: abcd = cd*(cd+1)/2 + ab
    return abcd

def twoe_integrals(e2):
    ERIraw = np.genfromtxt(e2,dtype=None)
    twoe = {eint(row[0],row[1],row[2],row[3]):row[4] for row in ERIraw}
    return twoe

def tei(a,b,c,d,twoe):
    return twoe.get(eint(a,b,c,d),0.0)

def make_tensor(twoe,t):
    dim = len(t)
    e2 = np.zeros((dim, dim, dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            for k in range(0, dim):
                for l in range(0, dim):
                    e2[i,j,k,l] = tei(i+1,j+1,k+1,l+1,twoe)
    return e2   

def make_density_matrix(orb,norb):
    dim,dim = orb.shape
    D = np.zeros((dim,dim))
    for i in range(norb):
        for j in range(dim):
            for k in range(dim):
                D[j,k] = D[j,k]+2*orb[j,i]*orb[k,i]
    return D

def fockJK(e2,D):
    dim,dim = D.shape
    JK = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    JK[i,j] = JK[i,j] + D[k,l]*(e2[i,j,k,l]-0.5*e2[i,l,k,j])
    return JK



# Start the code here

# If we work without nwchem, we must get the necessary matrix
Nelec= 10 #water 2*1+8
s = np.load('s.npy')
t = np.load('t.npy')
v = np.load('v.npy')
twoe = twoe_integrals('e2.dat')
e2 = make_tensor(twoe, t)
enuc = nuclear_repulsion('enuc.dat')

# HF code

# Step1: Form Transformation matrix fron A0 to orth basis
# In MATLAB, I think we could directly use s^-1/2, could we still do it in np?
e,u =np.linalg.eig(s) # e is the eigenvalues, not a matrix, which is different from MATLAB. u is the matrix of diagonalization
e=e**(-1./2)
s_12 = np.diag(e) # s_12 is the diagnol matrix we need but not the s matrix
X = np.matmul(u,np.matmul(s_12,u.T)) #X is the s matrix we need. X = u*s_12*u^-1

# Step2: Build the initial (guess) density
H = t + v # H does not contain two-electron integral. We take this H as the initial guess F0
F=np.matmul(X.T,np.matmul(H,X)) # F0'=X.T*F0*X, and here H=F beacuse we do not take two-electron into consideration.
eig, orb_o = np.linalg.eig(F) # eig is the enenrgy of t + v, and orb_o is the coeffiecnt matrix
idx = eig.argsort()[::1] 
eig_sort = eig[idx]
orb_o_sort = orb_o[:,idx]
orb = np.matmul(X,orb_o_sort) # orb is the final coeffeicent matrix we want. C0=X*C
norb = int(Nelec/2) # norb is the number of orbitals, we are doing with closed-shell
D = make_density_matrix(orb,norb)# D is the density matrix

# Step1 and Step2 actaully do the first SCF cycle to provide the initial matrix for Step3
# So basically most code of step3 are just the repeat of step1 and step2

# Step3: Start of SCF cycles
scf_steps = 100
e_old = np.sum(np.multiply(D,H)) # energy
D_old = D # density matrix

for iter in range(scf_steps):
    JK = fockJK(e2,D) #Two-electron intrgral matrix 
    F = H + JK # Real Fock matrix, contaning two-electron integral
    F_orth = np.matmul(X.T,np.matmul(F,X)) # F0'=X.T*F0*X
    eig,orb_o=np.linalg.eig(F_orth)
    idx = eig.argsort()[::1] 
    eig_sort = eig[idx]
    orb_o_sort = orb_o[:,idx]
    orb = np.matmul(X,orb_o_sort)
    D = make_density_matrix(orb,norb)
    
    energy = 0.5*np.sum(np.multiply(D,H+F)) + enuc # Total HF energy including nuclear repulsion
    rmsd = np.sum(np.power(np.power((D-D_old),2),0.5))
    print(iter,energy,rmsd)
    
    if abs(e_old - energy) < 1.e-8:	  #convergence criterion
        if rmsd < 1.e-8:	
            break
    D_old = D	
    e_old = energy
    
 # HF Done
 
 # MP2/3 code
 
 #Step1: Transform integrals
 
 
 
