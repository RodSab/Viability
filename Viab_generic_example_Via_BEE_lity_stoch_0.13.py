# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:48:10 2017
Last update 19 09 2023

@author: Rodolphe Sabatier, INRAE Ecodeveloppement

A clean/not too messy generic code to run basic viability related algorithms
It computes :
    - The viability kernel
    - Resilience bassins
    - Maps of adaptability and robustness

Works for up to 3 states and 5 controls

The model is illustrated here with a case study related to the sustainability of a bee farming system 
see Kouchner et al. for details on the case study:
 - Kouchner C. et al. (2019) Intégrer l’adaptabilité dans l’analyse de la durabilité des exploitations apicoles, Innovations Agronomiques (77)31-43
 https://hal.inrae.fr/hal-02900352/document


User should adjust the following sections to the case study:
    - General parameters
    - Case study related parameters
    - Case study related functions
    - Constraints
    - Dynamics of the system    
    
"""

import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mp
import pickle
import random
import math
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab


plt.close('all')
#mlab.close('all')


########################
## General parameters ##
########################
# Parameters of this section should be adjusted to the case study

path='D:\Home_rodolphe\Projets\Via_bee_lity\Save\\' # were to save the different viability objects
activ_viab=1 # if == 1, computes the viability algorithms
             # if == 0, skips this step and moves directly to the plotting functions 
additional_plotting=1 # activates some extra plotting functions (see around line 885 for details)


# Time related parameters that shall be adjusted to the case study
T=10 # Time horizon
Res_Horizon=T # Time horizon used for computing the resilience bassin
t0=0 # Initial time
dt=1 # Temporal discretization step 



# Parameters related to the discretization of States and controls, shall be adjusted to the case study 
nstates=2 # Number of state variables
nctrl=3 # Number of control variables
d_coeff=2 # Discretization coefficient, that simultaneously impacts the discretization of all states and controls

# State 1 ; Number of non productive swarms
x1min=0 # lower boundary of the discretization grid on x1
x1max=500 # upper boundary of the discretization grid on x1
x1d=20*d_coeff # discretization step on x1

# State 2 ; number of productive swarms
x2min=0
x2max=500
x2d=20*dt*d_coeff

# State 3
x3min=0
x3max=10
x3d=1*d_coeff

# Ctrl 1 ; Number of unproductive swarms created from productive ones
u1min=0
u1max=500
u1d=20*d_coeff

# Ctrl 2 ; Prop of queens allocated to x1
u2min=0
u2max=1
u2d=0.1*d_coeff

# Ctrl 3 ; Prop of queens allocated to newly divided swarns
u3min=0
u3max=1
u3d=0.1*d_coeff

# notice: 1-(u1+u2)= prop of queens allocated to old x0

# Ctrl 4
u4min=0
u4max=10
u4d=1*d_coeff

# Ctrl 5
u5min=0
u5max=10
u5d=1*d_coeff

# Vector of uncertainty values used
Omega=[0.9,1.0,1.1] # set to [1] if no uncertainty 


# computing a temporal parameter
len_t=int(((T-t0)/dt)+1) # do not modify 



###################################
## Case study related parameters ##
###################################

# Constraints
Income_thr=10000. # Constraint relative to the income
work_time_thr=700.
MinProdSwarns=20. # Constraint relative to the number of productive swarns

# Parameters needed to compute the dynamics of the case-study
S1=0.8 # base survival rate of the productive swarns
S0=0.9 # base survival rate of the unproductive swarns 
T01=0.5 # base transition rate from 0 to 1
T10=0.1 # base transition rate from 1 to 0
S0plus=0.5 # improvement in survival rate if 100% of x0 recieve new queens
S1plus=0.1 # improvement in survival rate if 100% of x1 recieve new queens
T01plus=0.3 # improvement in transition rate from 0 to 1 if 100% of x0 recieve new queens
T10plus=0.1 # diminution in regression rate from 1 to 0 if 100% of x1 recieve new queens
alpha=0.1 # proportion of x0 that are regrouped to make x1 (2 x0 = 1 x1)


beta=0.75 # threshold above which the production of new swarns leads to a regression of the prod swarns to unprod swarns
eps_1=0.1 # mortality rate of a newly divided swarn
#phy=100 # total nb of queens added
phyprop=0.5 # prop of the number of swarn that can produce queens


# Economic parameters
#Honey_sold=150.   # Honey sold per productive swarn
#eps_honey=0.3     # Proportion: Honey sold per x0 / honey sold per x1 
#Cost_u0=200.      # Hypothetical cost associated to the division of a x1
#Cost_x=40.      # Hypothetical cost associated to the aintenance of a colony

Q_h=20
Xi_h=6
Xi_u0=2
Xi_x=20
eps_x=0.8
Xi_n=0.5
Xi_q=20
Xi_s=2000
eps_h=0
Xi_qb=0 # Queen rearing structural costs
Xi_w=0 # Total wages


# Working time related parameters
tau_0=0.0625
tau_d=0.125
n_n=30
tau_r=0.0625
n_r=30
tau_q=0.05
tau_x=0.5
tau_1=0.2
tau_e=0.0017
tau_s=15


##################################
## Case study related functions ##
##################################

# Functions needed to compute the constraints of the system
def income(xx,uu,oom,t):
    p=Q_h*Xi_h*(xx[1]+eps_h*xx[0])-uu[0]*Xi_u0-Xi_x*(eps_x*xx[0]+xx[1])-(f_nx1(xx,uu,oom)+f_t01(xx,uu,oom))*Xi_n-phy(xx[0]+xx[1])*Xi_q-Xi_s-Xi_qb-Xi_w# honey sold plus unproductive swarms sold minus production cost of queens(10 are free)   
    return p
    
def work_time(xx,uu,t):
    p=tau_0*uu[0]+tau_d*math.ceil(uu[0]/n_n)+phy(xx[0]+xx[1])*(1-uu[2])*tau_r+tau_d*math.ceil((phy(xx[0]+xx[1])*(1-uu[2])/n_r))+tau_q*phy(xx[0]+xx[1])+tau_x*(xx[0]+xx[1])+tau_1*xx[1]+tau_e*Xi_h*(xx[1]+eps_h*xx[0])+tau_s
    return p
    
###############################################
## Constraints (specific to the case study) ##
###############################################

# write one function per constraint
# each function shall take xx and uu as parameters 
# each function returns 0 if the contraint is respected and 1 if it is not
# adjust "nphi" to the number of constraints
# each function should be named "phi_n" for n ranging from 1 to nphi

nphi=5 # total number of constraints considered (number of phi_n functions)

# Start with the constraints only refering to states that shall be respected at t=T
def phi_1(xx,uu,oom,t):
    if xx[1]>=MinProdSwarns: # minimum number of productive swarms
        temp=0
    else:
        temp=1
    return temp
nphi_states=1 # number of constraints relative to the final state

# Then define the constraints combining states and controls or that don't need to be respected at t=T
def phi_2(xx,uu,oom,t): # income should be above a threshold
    p=income(xx,uu,oom,t) 
    if p>Income_thr:
        temp=0
    else:
        temp=1
    return temp
    
def phi_3(xx,uu,oom,t): # can't produce more than 2 unproductive swarm per productive swarm, only on 75% of the productive swarms
    if uu[0]<2*beta*xx[1]:
        temp=0
    else:
        temp=1
    return temp
    
def phi_4(xx,uu,oom,t): # proportion of queens added to the different classes = 1
    if uu[1]+uu[2]<=1:
        temp=0
    else:
        temp=1
    return temp
    
def phi_5(xx,uu,oom,t): # Working time should be bellow a certain threshold
    p=work_time(xx,uu,t) 
    if p<=work_time_thr:
        temp=0
    else:
        temp=1
    return temp


########################
## Plotting functions ##
########################
# Not need to adapt to the case study

def Save_Pick(name,pth,file_name):
    outfile=open(str(pth)+'Save_'+str(file_name)+'.p','wb')
    pickle.dump(name,outfile)
    outfile.close() 

def envlp (mat):
    l0=len(mat[:,1,1])
    l1=len(mat[1,:,1])
    l2=len(mat[1,1,:])
    #print(l0,l1,l2)    
    ev_mat=numpy.zeros((l0+1,l1+1,l2+1))
    for jj in range(l0-2):
        for kk in range(l1-2):
            for ll in range(l2-2):
                j=jj+1
                k=kk+1
                l=ll+1
                if mat[j,k,l]==1: 
                    if mat[j+1,k,l]==1 and mat[j-1,k,l]==1 and mat[j,k+1,l]==1 and mat[j,k-1,l]==1 and mat[j,k,l+1]==1 and mat[j,k,l-1]==1:
                        ev_mat[j,k,l]=0
                    else: 
                        ev_mat[j,k,l]=1
                    if j==1:
                        ev_mat[j,k,l]=1
                    if k==1:
                        ev_mat[j,k,l]=1
                    if l==1:
                        ev_mat[j,k,l]=1
                else: 
                    ev_mat[j,k,l]==0
    return ev_mat



def Tube(mat):
    #Tube_plot=numpy.zeros((len(mat[:,0,0]),len(mat[0,:,0]),len(mat[0,0,:])))    
    Tube_plot=numpy.zeros((len(mat[:,0,0])+1,len(mat[0,:,0])+1,len(mat[0,0,:])+1))    
    for i in range(len(mat[:,0,0])):
        for j in range(len(mat[0,:,0])):
            for k in range(len(mat[0,0,:])):
                if mat[i,j,k]==0:
                    Tube_plot[i,j,k]=1
                    #Jprint(i,j,k,ttt)
                else:
                    Tube_plot[i,j,k]=0
    return Tube_plot


def xyz(mat):
    x3D=[]
    y3D=[]
    z3D=[]
    TB=Tube(mat)
    env=envlp(TB) 
    #print(env)
    
    for i in range(len(mat[:,0,0])+1):
        for j in range(len(mat[0,:,0])+1):
            for l in range(len(mat[0,0,:])+1):
                #print(Tube_XU[i,j,l,t_proj])
                if env[i,j,l]==1:
                    x3D.append(i)
                    y3D.append(j)
                    z3D.append(l)
    return x3D, y3D, z3D
    
###########################################
## Viability algorithm related functions ##
###########################################
# No need to adapt to the case study


def grid(d,zmin,zmax):
    # A function that creates a grid discretizing a state (or a control) space
    # the grid is a list of values (not a list of indices)
    # arguments are : 
    # d: the discretization step,
    # zmin: the value of the first element of the grid 
    # zmax: the value of the last element of the grid  
    gridz=[]
    z=zmin
    while z<zmax+d:
        gridz.append(z)
        z+=d
    return gridz

def proj(z,gz):
    # A function that transforms a value into its projection on a grid and gives the index
    # the projection is made to the closest value of the grid (with "round", not with "int")
    # the input is any value, the output are one of the values of the grid and the related index
    # arguments are : 
    # z: the initial value to project,
    # gz: the grid on which we project the value
    if len(gz)>1: 
        zz=float(min(z,gz[-1]))
        zz=max(gz[0],zz)    
        iz=int(round((zz-gz[0])/(gz[1]-gz[0])))
    else:
        iz=0
    return iz,gz[iz] # it returns the index and the value of the projection on the grid

def proj_v(v):
    vI=[proj(v[0],gridx1)[0],proj(v[1],gridx2)[0],proj(v[2],gridx3)[0]]
    return vI
    
def cont(gz,iz):
    # A function that transforms an index into the corresponding element of a grid
    return gz[iz]
    
def viab_stctrl(xx,uu,oom,t):
    # This function verifies if the state control-combination respects the constraints at time t
    # it returns 0 when viable    
    ph=0
    for n in range(nphi):
        p_temp=globals()['phi_'+str(n+1)](xx,uu,oom,t)
#        if p_temp==1:
#            print('***')
        ph+=p_temp
    return ph
        
def viab_final_st(xx,uu,om,t):
    # This function verifies if the state respects the constraints at time t=T
    # it returns 0 when viable
    phi=0
    for n in range(nphi_states):
        phi+=globals()['phi_'+str(n+1)](xx,uu,om,t)
    return phi        
        
def vect_state_ctrl(xx1,xx2,xx3,uu1,uu2,uu3,uu4,uu5):
    vx=[xx1,xx2,xx3]
    vu=[uu1,uu2,uu3,uu4,uu5]
    return vx,vu
    
def viab(mat,xx,t):
    v=numpy.amin(mat[xx[0],xx[1],xx[2],:,:,:,:,:,t])
    return v
    
def fill_value(v_a,met,xx,uu,t):
    v_a[xx[0],xx[1],xx[2],uu[0],uu[1],uu[2],uu[3],uu[4],t]=met
            
def Ker(v_a):
    kk=numpy.zeros((len(gridx1),len(gridx2),len(gridx3),len_t))
    for i1 in range(len(gridx1)):
        for i2 in range(len(gridx2)):
            for i3 in range(len(gridx3)):
                for tt in range(len_t):
                    kk[i1,i2,i3,tt]=min(1,numpy.amin(v_a[i1,i2,i3,:,:,:,:,:,tt])) 
    return kk



############################
## Dynamics of the system ##
############################
# A series of functions specific to the case study
# The key function is "dynamics"
# it is the one used by the viability algorithms, adjust it to the case study without changing it's outputs

def phy(xx):
    p=xx*phyprop
    return p

def f_S0(xx,uu,oom):
    # this function accounts for:
    #   - base survival of x0
    #   - effect of adding queens to x0
    #   - effect of regrouping x0 to make new x1
    xt=xx[0]+xx[1]
    if xx[0]!=0:
        s=max(0,min(1,oom*S0+S0plus*((1-uu[1]-uu[2])*phy(xt)/xx[0])-0.5*alpha)) # base survival (S0) plus improvement due to added queens (S0plus) minus disparition of 0.5 alpha (the other 0.5 alpha is counted in T_01)
    else:
        s=max(0,oom*S0-0.5*alpha)
    return s # it's a proportion
    
def f_S1(xx,uu,oom):
    # this function accounts for:
    #   - base survival of x1
    #   - effect of adding queens to x1
    xt=xx[0]+xx[1]
    if xx[1]!=0:
        s=min(1,oom*S1+S1plus*(uu[1]*phy(xt))/xx[1]) # base survival (S1) plus improvement due to added queens (S1plus)
    else:
        s=0
    return s # it's a proportion

def f_t01(xx,uu,oom):
    # this function accounts for:
    #   - base growth of colonies x0
    #   - effect of adding queens to all x0 on this growth
    #   - effect of regrouping x0 to make new x1
    xt=xx[0]+xx[1]
    if xx[0]!=0:
        trans=min(1,oom*T01+T10plus*((1-uu[1]-uu[2])*phy(xt))/xx[0]+0.5*alpha)
    else:
        trans=0
    return trans # it's a proportion
    
def f_t10(xx,uu,oom):
    # this function accounts for:
    #   - base regression of colonies x1 to x0
    #   - effect of adding queens to all x1
    #   - effect of overdividing swarns x1
    xt=xx[0]+xx[1]
    if xx[1]!=0:
        if uu[0]<beta*xx[1]: 
            car=0
        else:
            car=1
        trans=max(0,min(1,oom*T10-T01plus*uu[1]*phy(xt)/xx[1]+car*uu[0]/(xx[1]-beta))) # base regression from x1 to x0 - diminition of this coeff by adding queens + shift from 1 to 0 due to over dividing x1
    else:
        trans=0
    return trans # it's a proportion

def f_nx0(xx,uu,oom):
    # this function accounts for:
    #   - creation of x0 by dividing x1 - including a proportion of success (1-eps1)
    #   - effect of adding queens to newly created x0 that reduces the number of new x0 (they become x1 directly)
    xt=xx[0]+xx[1]
    if uu[0]!=0:
        new=(1-eps_1)*(1-min(1,(uu[2]*phy(xt))/uu[0])) 
    else:
        new=0
    return new # it's a proportion

def f_nx1(xx,uu,oom):
    # this function accounts for:
    #   - creation of x1 by dividing x1 and adding queens directly to these new swarns - including a proportion of success (1-eps1)
    # notice, eps1 is supposed the same as in f_nx0
    xt=xx[0]+xx[1]
    if uu[0]!=0:
        new=(1-eps_1)*(min(1,(uu[2]*phy(xt))/uu[0])) 
    else:
        new=0
    return new # it's a proportion
    
def dynamics(xx,uu,oom,t):
    # this fucntion reflects the state-control dynamics of the system
    # arguments are 
    # xx: a list of state values, xx=[x1,...,xn]
    # uu: a list of ctrl values, uu=[u1,...,un]
    # t : the time step
    # output is a vector of states at t+1, xx_plusone=[x1[t+1],...,xn[t+1]]
    xx_plusone=[[0],[0],[0]]
    
    
    # Bee population model
    # second version of the bee dynamic model after the meeting of october 2017   
    T_x0_x1=f_t01(xx,uu,oom) 
    T_x1_x0=f_t10(xx,uu,oom) 
    S_x1=f_S1(xx,uu,oom)
    S_x0=f_S0(xx,uu,oom)
    N_x0=f_nx1(xx,uu,oom)
    N_x1=f_nx1(xx,uu,oom)
    
    xx_plusone[0]=S_x0*xx[0]+T_x1_x0*xx[1]-T_x0_x1*xx[0]+N_x0*uu[0]
    xx_plusone[1]=S_x1*xx[1]+T_x0_x1*xx[0]-T_x1_x0*xx[1]+N_x1*uu[0]
    
    xx_plusone[0]=max(min(xx_plusone[0],x1max),x1min)
    xx_plusone[1]=max(min(xx_plusone[1],x2max),x2min)
    return xx_plusone
    

#########################
## Viability algorithm ##
#########################

print('Initialization...')

# Initialization of the grids
gridx1=[0]
gridx2=[0]
gridx3=[0]
gridu1=[0]
gridu2=[0]
gridu3=[0]
gridu4=[0]
gridu5=[0]
gridx1=grid(x1d,x1min,x1max)
if nstates>1:
    gridx2=grid(x2d,x2min,x2max)
    if nstates>2:
        gridx3=grid(x3d,x3min,x3max)
gridu1=grid(u1d,u1min,u1max)
if nctrl>1:
    gridu2=grid(u2d,u2min,u2max)
    if nctrl>2:
        gridu3=grid(u3d,u3min,u3max)
        if nctrl>3:
            gridu4=grid(u4d,u4min,u4max)
            if nctrl>4:
                gridu5=grid(u5d,u5min,u5max)
        
# Initiailzation of the Value matrix at time t=Horizon        

if activ_viab==1:    

    value_adapt=numpy.zeros((len(gridx1),len(gridx2),len(gridx3),len(gridu1),len(gridu2),len(gridu3),len(gridu4),len(gridu5),len_t))
    value_stoch=numpy.zeros((len(gridx1),len(gridx2),len(gridx3),len(gridu1),len(gridu2),len(gridu3),len(gridu4),len(gridu5),len_t))
    value_res=numpy.zeros((len(gridx1),len(gridx2),len(gridx3)))
    n_u_viab=numpy.zeros((len(gridx1),len(gridx2),len(gridx3),len_t))
    
    
    
    for i1,x1 in enumerate(gridx1):
        for i2,x2 in enumerate(gridx2):
            for i3,x3 in enumerate(gridx3):
                vtp=[]
                vtp_stoch=0
                for om in Omega:
                    vtp.append(viab_final_st([x1,x2,x3],[],om,len_t-1))
                    vtp_stoch+=viab_final_st([x1,x2,x3],[],om,len_t-1)
                value_adapt[i1,i2,i3,:,:,:,:,:,len_t-1]=max(vtp)
                value_stoch[i1,i2,i3,:,:,:,:,:,len_t-1]=min(1,float(vtp_stoch)/len(Omega))
                value_res[i1,i2,i3]=Res_Horizon+1
                n_u_viab[i1,i2,i3,len_t-1]=100
                
    # Computing viability
    print('Computing the viability kernel...')
    
    
    
    for ttt in reversed(range(((T-t0)/dt))):
    #for ttt in range(len_t-1):
        t=dt*ttt+t0
        print(ttt)
        for i1,x1 in enumerate(gridx1):
            for i2,x2 in enumerate(gridx2):
                for i3,x3 in enumerate(gridx3):
                    for j1,u1 in enumerate(gridu1):
                        for j2,u2 in enumerate(gridu2):
                            for j3,u3 in enumerate(gridu3):
                                for j4,u4 in enumerate(gridu4):
                                    for j5,u5 in enumerate(gridu5):
                                        metric_viab=0
                                        I,J=vect_state_ctrl(i1,i2,i3,j1,j2,j3,j4,j5)
                                        v_m=[]
                                        phi_om=0
                                        for k,om in enumerate(Omega):
                                            phi=0
                                            X,U=vect_state_ctrl(x1,x2,x3,u1,u2,u3,u4,u5)
                                            for tt in range(dt):                                                    
                                                phi+=viab_stctrl(X,U,om,t+tt)
                                                X=dynamics(X,U,om,t+tt)
                                                Iplus=proj_v(X)
                                                # the following lines test if the next state is adaptable
                                                if n_u_viab[i1,i2,i3,ttt+1]>=5:
                                                    viab_flex=0
                                                else:
                                                    viab_flex=1
                                                phi+=viab_flex
                                            v_m.append(max(metric_viab,phi+viab(value_adapt,Iplus,ttt+1)))
                                            phi_om+=min(1,phi+viab(value_stoch,Iplus,ttt+1))
                                        fill_value(value_stoch,float(phi_om)/len(Omega),I,J,ttt)
                                        fill_value(value_adapt,max(v_m),I,J,ttt)
                                        if value_adapt[i1,i2,i3,j1,j2,j3,j4,j5,ttt]==0:
                                            n_u_viab[i1,i2,i3,ttt]+=1
        
    print('Computing the resilience basin')
    for i1,x1 in enumerate(gridx1):
        for i2,x2 in enumerate(gridx2):
            for i3,x3 in enumerate(gridx3):                        
                if numpy.amin(value_adapt[i1,i2,i3,:,:,:,:,:,0])==0:
                    value_res[i1,i2,i3]=0
                else:
                    for ttt in range(Res_Horizon):
                        t=dt*ttt+t0
                        #print(ttt)
                        res=20
                        for j1,u1 in enumerate(gridu1):
                            for j2,u2 in enumerate(gridu2):
                                for j3,u3 in enumerate(gridu3):
                                    for j4,u4 in enumerate(gridu4):
                                        for j5,u5 in enumerate(gridu5):
                                            metric_viab=0
                                            I,J=vect_state_ctrl(i1,i2,i3,j1,j2,j3,j4,j5)
                                            v_m=[]
                                            v_res=[]
                                            for k,om in enumerate(Omega):
                                                X,U=vect_state_ctrl(x1,x2,x3,u1,u2,u3,u4,u5)
                                                for tt in range(dt):
                                                    phi+=viab_stctrl(X,U,om,t+tt)
                                                    X=dynamics(X,U,om,t+tt)
                                                    Iplus=proj_v(X)
                                                v_res.append(1+value_res[Iplus[0],Iplus[1],Iplus[2]])
                                            res=min(res,max(v_res))
                        value_res[i1,i2,i3]=res
        
    Kernel=Ker(value_adapt)
    if numpy.amin(Kernel[:,:,:,0])==0:
        print('Viability Kernel is not empty')
        #print(Kernel[:,:,0,0])
        print('Saving....')
        Save_Pick(value_adapt,path,'Value_adapt')
        Save_Pick(value_stoch,path,'Value_stoch')
        Save_Pick(value_res,path,'Value_res')
    else:
        print('Viability Kernel is empty')
    
print('loading...')    
value_adapt=[]
infile=open(str(path)+'Save_Value_adapt.p','rb')
value_adapt=pickle.load(infile)  
infile.close()   
Kernel=Ker(value_adapt)

value_stoch=[]
infile=open(str(path)+'Save_Value_stoch.p','rb')
value_stoch=pickle.load(infile)  
infile.close()   
Robustness=Ker(value_stoch)

value_res=[]
infile=open(str(path)+'Save_Value_res.p','rb')
value_res=pickle.load(infile)  
infile.close()   
Resilience=value_res

if numpy.amin(Kernel[:,:,:,0])!=0:
    print('Viability Kernel is empty')
else:


    Map_Adapt_deter=numpy.zeros((len(gridx1),len(gridx2),len(gridx3)))
    Map_Adapt_stoch=numpy.zeros((len(gridx1),len(gridx2),len(gridx3)))
    
    for i1 in range(len(gridx1)):
        for i2 in range(len(gridx2)):
            for i3 in range(len(gridx3)):
                for j1,u1 in enumerate(gridu1):
                    for j2,u2 in enumerate(gridu2):
                        for j3,u3 in enumerate(gridu3):
                            for j4,u4 in enumerate(gridu4):
                                for j5,u5 in enumerate(gridu5):
                                    if value_adapt[i1,i2,i3,j1,j2,j3,j4,j5,0]==0:
                                        Map_Adapt_deter[i1,i2,i3]+=1
                                    Map_Adapt_stoch[i1,i2,i3]+=(1-value_stoch[i1,i2,i3,j1,j2,j3,j4,j5,0])    

    if nstates==2:
        
        eps=0.01
        red=[(0.0,  0.0, 0.0)]
                   
        ncol=int(10*numpy.amax(Robustness[:,:,0,0]))                 
        for i in range (ncol):
            red.append((min(1.0,(i*1./(ncol+1)+eps)),0.0,1.0-i*(1./ncol)))
        red.append((1.0,0.0,0.0))
        
#        print('red')
#        print(red)

        cdict = {'red':   red,

         'green': ((0.000,  0.6, 0.6),
                   (0.0+eps,  0.6, 0.0),
                   (1.0,  0.0, 0.0)),


         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}
                   
        CustomMap = mp.colors.LinearSegmentedColormap('CustomMap', cdict)
        
        
        plt.figure(1)
        plt.matshow(Robustness[:,:,0,0].T,cmap=CustomMap)
        plt.title('Robustness')
        plt.xlabel('Productive swarms (x10)')
        plt.ylabel('Unproductive swarms (x10)')
        
        eps=0.01
        red=[(0.0,  0.0, 0.0)]
                   
        ncol=int(10*numpy.amax(Resilience[:,:,0]))                 
        for i in range (ncol):
            red.append((min(1.0,(i*1./(ncol+1)+eps)),0.0,1.0-i*(1./ncol)))
        red.append((1.0,0.0,0.0))
        
#        print('red')
#        print(red)

        cdict = {'red':   red,

         'green': ((0.000,  0.6, 0.6),
                   (0.0+eps,  0.6, 0.0),
                   (1.0,  0.0, 0.0)),


         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}
                   
        CustomMap = mp.colors.LinearSegmentedColormap('CustomMap', cdict)
        
        plt.figure(2)
        plt.matshow(Resilience[:,:,0].T,cmap=CustomMap)
        plt.title('Resilience')
        plt.xlabel('Productive swarms (x10)')
        plt.ylabel('Unproductive swarms (x10)')
        
        X3D,Y3D,Z3D=xyz(Kernel[:,:,0,:])
#        print(X3D)
#        print(Y3D)
#        print(Z3D)
        fig=mlab.figure(3)
        pts = mlab.points3d(Z3D,X3D,Y3D, Z3D, scale_mode='none',scale_factor=1,mode='cube')
        mesh = mlab.pipeline.delaunay2d(pts)
        mlab.axes(extent=[0,len_t+2,0,len(gridx1)+2,0,len(gridx2)+2],nb_labels=5,xlabel='Time',ylabel='Unproductive swarms',zlabel='Productive swarms')

        
        
        cdict = {'red':   ((0.000,  0.0, 0.0),
                   (0.0+eps,  0.0, 0.1),
                   (1.0,  1.0, 0.0)),

         'green': ((0.000,  0.0, 0.0),
                   (0.0+eps,  0.0, 0.2),
                   (1.0,  0.2, 0.0)),


         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}
                   
        CustomMap2 = mp.colors.LinearSegmentedColormap('CustomMap2', cdict)        
        
        #print(Map_Adapt_deter[:,:,0])
        #print(Map_Adapt_stoch[:,:,0])
        
        plt.figure(10)
        plt.matshow(Map_Adapt_deter[:,:,0].T,cmap=CustomMap2)
        plt.title('Adaptivity (det)')
        plt.xlabel('Productive swarms (x10)')
        plt.ylabel('Unproductive swarms (x10)')
        
        plt.figure(11)
        plt.matshow(Map_Adapt_stoch[:,:,0].T,cmap=CustomMap2)
        plt.title('Adaptivity (stoch)')
        plt.xlabel('Productive swarms (x10)')
        plt.ylabel('Unproductive swarms (x10)')
    
    v=1
    while v>0:      
#        print('*')
        X=[random.choice(gridx1),random.choice(gridx2),random.choice(gridx3)]
        I=proj_v(X)
        v=Kernel[I[0],I[1],I[2],0]
#        print(v)
    trajectories=[[X[0]],[X[1]],[X[2]],[],[],[],[],[],[]]
    v_omega=[]    
    
    for ttt in range(((T-t0)/dt)):
#        print('*')
        t=dt*ttt+t0        
        om=random.choice(Omega)
        X=[trajectories[0][-1],trajectories[1][-1],trajectories[2][-1]]
        I=proj_v(X)
        v=1
        while v>0:       
            U=[random.choice(gridu1),random.choice(gridu2),random.choice(gridu3),random.choice(gridu4),random.choice(gridu5)]
            J=[proj(U[0],gridu1),proj(U[1],gridu2),proj(U[2],gridu3),proj(U[3],gridu4),proj(U[4],gridu5)]
            phi=0
            temp_inc=0
            for tt in range(dt):
                phi+=viab_stctrl(X,U,om,t+tt)
                temp_inc+=income(X,U,om,t)
                X=dynamics(X,U,om,t+tt)# Warning make it an intermediate variable Xplus but do not replace X !!!
            Iplus=proj_v(X)
            v=phi+viab(value_adapt,Iplus,ttt+1)
        trajectories[0].append(X[0])
        trajectories[1].append(X[1])
        trajectories[2].append(X[2])
        trajectories[3].append(U[0])
        trajectories[4].append(U[1]*phy(X[0]+X[1]))
        trajectories[5].append(U[2]*phy(X[0]+X[1]))
        trajectories[6].append(phy(X[0]+X[1])*(1-U[1]-U[2]))
        trajectories[7].append(U[4])
        trajectories[8].append(temp_inc)
        v_omega.append(om)
        
    plt.figure(4)    
    plotx1,=plt.plot(trajectories[0])
    plt.figure(4)  
    plotx2,=plt.plot(trajectories[1])
    plt.axhline(linestyle='--',color='g',y=MinProdSwarns)
    plt.ylim(min(x1min,x2min),max(x1max,x2max)*1.1)
    plt.legend([plotx1,plotx2],['Unprod', 'Prod'])
    #plt.plot(trajectories[2])
    
    plt.figure(5)    
    plotu1,=plt.plot(trajectories[3])
    plt.figure(5)
    plotu2,=plt.plot(trajectories[4])
    plt.figure(5)    
    plotu3,=plt.plot(trajectories[5])    
    plt.figure(5)    
    plotu4,=plt.plot(trajectories[6])  
    plt.legend([plotu1,plotu2,plotu3,plotu4],['New Unprod', 'Queens added in prod', 'Queen added in new unprod', 'Queen added in old unprod'])
    plt.ylim(min(u1min,u2min,u3min),max(u1max,phy(x1max+x2max)*1.3))
    
    plt.figure(6)    
    plotu1,=plt.plot(trajectories[8])  
    plt.legend([plotu1],['Income'])
    plt.axhline(linestyle='--',color='b',y=Income_thr)
    plt.ylim(0,max(trajectories[8])*1.1)
    
    #plt.plot(trajectories[6])
    #plt.plot(trajectories[7])
    #plt.plot(trajectories[8])

#    print(trajectories[0])
#    print(trajectories[1])
#    print(trajectories[3])
#    print(trajectories[4])
#    print(trajectories[5])
    
    
    ptx1=len(gridx1)/9
    ptx2=len(gridx2)/9
    Xinit=[[ptx1,ptx2,0],[ptx1,2*ptx2,0],[ptx1,3*ptx2,0],[ptx1,4*ptx2,0],[ptx1,5*ptx2,0],[ptx1,6*ptx2,0],[ptx1,7*ptx2,0],[ptx1,8*ptx2,0],
            [2*ptx1,ptx2,0],[2*ptx1,2*ptx2,0],[2*ptx1,3*ptx2,0],[2*ptx1,4*ptx2,0],[2*ptx1,5*ptx2,0],[2*ptx1,6*ptx2,0],[2*ptx1,7*ptx2,0],[2*ptx1,8*ptx2,0],
            [3*ptx1,ptx2,0],[3*ptx1,2*ptx2,0],[3*ptx1,3*ptx2,0],[3*ptx1,4*ptx2,0],[3*ptx1,5*ptx2,0],[3*ptx1,6*ptx2,0],[3*ptx1,7*ptx2,0],[3*ptx1,8*ptx2,0],
            [4*ptx1,ptx2,0],[4*ptx1,2*ptx2,0],[4*ptx1,3*ptx2,0],[4*ptx1,4*ptx2,0],[4*ptx1,5*ptx2,0],[4*ptx1,6*ptx2,0],[4*ptx1,7*ptx2,0],[4*ptx1,8*ptx2,0],
            [5*ptx1,ptx2,0],[5*ptx1,2*ptx2,0],[5*ptx1,3*ptx2,0],[5*ptx1,4*ptx2,0],[5*ptx1,5*ptx2,0],[5*ptx1,6*ptx2,0],[5*ptx1,7*ptx2,0],[5*ptx1,8*ptx2,0],
            [6*ptx1,ptx2,0],[6*ptx1,2*ptx2,0],[6*ptx1,3*ptx2,0],[6*ptx1,4*ptx2,0],[6*ptx1,5*ptx2,0],[6*ptx1,6*ptx2,0],[6*ptx1,7*ptx2,0],[6*ptx1,8*ptx2,0],
            [7*ptx1,ptx2,0],[7*ptx1,2*ptx2,0],[7*ptx1,3*ptx2,0],[7*ptx1,4*ptx2,0],[7*ptx1,5*ptx2,0],[7*ptx1,6*ptx2,0],[7*ptx1,7*ptx2,0],[7*ptx1,8*ptx2,0],
            [8*ptx1,ptx2,0],[8*ptx1,2*ptx2,0],[8*ptx1,3*ptx2,0],[8*ptx1,4*ptx2,0],[8*ptx1,5*ptx2,0],[8*ptx1,6*ptx2,0],[8*ptx1,7*ptx2,0],[8*ptx1,8*ptx2,0]]
    #Xinit=[[10,10,0]]
            
    img=[]
    vX3D=[]
    minx=1000       
    maxx=0
    miny=1000
    maxy=0
    minz=1000
    maxz=0
    
    if additional_plotting==1:
        for X0 in Xinit:
            X3D,Y3D,Z3D=xyz(value_adapt[X0[0],X0[1],X0[2],:,:,:,0,0,0])
    #        print(X3D)
    #        print(Y3D)
    #        print(Z3D)
            if X3D!=[]:
                vX3D.append([X3D,Y3D,Z3D])
                minx=min(minx,min(X3D))        
                maxx=max(maxx,max(X3D))
                miny=min(miny,min(Y3D))
                maxy=max(maxy,max(Y3D))
                minz=min(minz,min(Z3D))
                maxz=max(maxz,max(Z3D))
            else:
                vX3D.append([[],[],[]])
        
        for i in range (len(Xinit)):
            [X3D,Y3D,Z3D]=vX3D[i]
            fig=mlab.figure(7)
            pts = mlab.points3d(Z3D,X3D,Y3D, Z3D, scale_mode='none',scale_factor=1,mode='cube',vmin=minz, vmax=maxz,colormap='PuOr')
            mesh = mlab.pipeline.delaunay2d(pts)
            mlab.axes(extent=[minz,maxz+2,minx,maxx+2,miny,maxy+2],nb_labels=5,xlabel='U3',ylabel='U1',zlabel='U2')
            mlab.view(azimuth=30, elevation=60, distance=30)      
            img.append(mlab.screenshot())        
            mlab.close()
        
        plt.figure(8,(12,12))
        for i in range (len(Xinit)):
            f2=plt.subplot(8,8,i+1)
    #        print(Xinit[i])
            plt.title('x0='+str(cont(gridx1,Xinit[i][0]))+' x1='+str(cont(gridx2,Xinit[i][1])),fontsize=8)
            f2.imshow(img[i])
            f2.set_axis_off()
        plt.tight_layout()
    
    
    
    
    
    i=0
    plt.figure(20,(12,12))
    for X0 in Xinit:
        f2=plt.subplot(8,8,i+1)
        n_u_u1=[]
        for U1 in range(len(gridu1)):
            tu1=0
            for U2 in range(len(gridu2)):
                for U3 in range(len(gridu3)):
                    if value_adapt[X0[0],X0[1],X0[2],U1,U2,U3,0,0,0]==0:
                        tu1+=1
            n_u_u1.append(float(tu1)/(len(gridu2)*len(gridu3)))
        plt.title('x0='+str(int(cont(gridx1,Xinit[i][0])))+' x1='+str(int(cont(gridx2,Xinit[i][1]))),fontsize=8)
#        plt.ylim(0,30)
#        plt.xlim(0,u1max)
        i+=1      
        plt.bar(gridu1,n_u_u1,width=u1d)
        plt.xticks(rotation=60,fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim(0,1)
    plt.tight_layout()
    
    i=0    
    plt.figure(21,(12,12))
    for X0 in Xinit:
        f2=plt.subplot(8,8,i+1)
        n_u_u2=[]
        for U2 in range(len(gridu2)):
            tu2=0
            for U1 in range(len(gridu1)):
                for U3 in range(len(gridu3)):
                    if value_adapt[X0[0],X0[1],X0[2],U1,U2,U3,0,0,0]==0:
                        tu2+=1
            n_u_u2.append(float(tu2)/(len(gridu1)*len(gridu3)))
        plt.title('x0='+str(int(cont(gridx1,Xinit[i][0])))+' x1='+str(int(cont(gridx2,Xinit[i][1]))),fontsize=8)
        plt.ylim(0,30)
        plt.xlim(0,u2max)
        i+=1
        plt.bar(gridu2,n_u_u2,width=u2d)
        plt.xticks(rotation=60,fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim(0,1)
    plt.tight_layout()
    
    i=0            
    plt.figure(22,(12,12))
    for X0 in Xinit:
        f2=plt.subplot(8,8,i+1)
        n_u_u3=[]
        for U3 in range(len(gridu3)):
            tu3=0
            for U2 in range(len(gridu2)):
                for U1 in range(len(gridu1)):
                    if value_adapt[X0[0],X0[1],X0[2],U1,U2,U3,0,0,0]==0:
                        tu3+=1
            n_u_u3.append(float(tu3)/(len(gridu1)*len(gridu2)))
        plt.title('x0='+str(int(cont(gridx1,Xinit[i][0])))+' x1='+str(int(cont(gridx2,Xinit[i][1]))),fontsize=8)
        plt.ylim(0,30)
        plt.xlim(0,u3max)
        i+=1
        plt.bar(gridu3,n_u_u3,width=u3d)
        plt.xticks(rotation=60,fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylim(0,1)
    plt.tight_layout()

    plt.show()

    print('done')