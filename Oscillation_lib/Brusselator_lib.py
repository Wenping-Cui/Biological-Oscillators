from scipy.integrate import odeint
from scipy import optimize
from scipy import signal
import time
import seaborn
from Oscillation_lib.Brusselator_cython import Brusselator_loopProgress
from Oscillation_lib.Brusselator_cython import Brusse_Barato_loopProgress
from scipy.integrate import odeint
import pylab as plt
import numpy as np
import time
import imp
import os
import pandas as pd
#pdb.set_trace()
##########################################################################


class Brusselator_simulation(object):
    def __init__(self, par):
        self.v=par[0]
        self.mu=par[1];
        self.t_scale=par[2]
        self.k1=par[3]
        self.k_1=par[4]
        self.k2=par[5]
        self.k_2=par[6]
        self.k3=par[7]
        self.k_3=par[8]
        self.a = par[9]
        self.b = par[10]
###################################################################### 
# determinstic values      
        self.entropy=0;
        self.period=0;
        self.x, self.y=0, 0
        self.Y0 = [self.x, self.y]
        self.step=10.0
        self.Duration =2000.0
        self.path=os.getcwd()
        self.filename1 = 'Brusselator_data/Brusselator_'+'Data_mu_'+str(self.mu)+'_size_'+str(self.v)+'_.csv'
        self.filename1 = os.path.join(self.path,self.filename1)
        self.filename2 = 'Brusselator_data//Brusselator_'+'FPT_'+'mu_'+str(self.mu)+'_size_'+str(self.v)+'_.csv'
        self.filename2 = os.path.join(self.path,self.filename2)
        self.filename3 = 'Brusselator_data//Brusselator_'+'FPTDirect_'+'mu_'+str(self.mu)+'_size_'+str(self.v)+'_.csv'
        self.filename3 = os.path.join(self.path,self.filename3)
        self.numberofrealisations=int(self.Duration*self.step);
        self.dermin_period=0
        self.dermin_entropy=0;
        self.window_period=0
        self.D=0
        self.tauc=0
        self.varT=0
        self.autocorrelation_period=0
        self.win_threshold1, self.win_threshold2=0,0;
        self.par = np.array([self.a, self.b, self.k1*self.t_scale, self.k2*self.t_scale, self.k3*self.t_scale, self.k_1*self.t_scale, self.k_2*self.t_scale, self.k_3*self.t_scale])
########################################################################
    def determinstic_simulation(self,plot=False, plot_range=0.1, threslold_x=0.25):
        t = np.linspace(0, self.Duration, self.step*self.Duration*30)
        sol = odeint(self.Brusse_Barato, self.Y0, t, args=(self.par,))
        X_t, Y_t=sol[:,0],sol[:,1]
        current_d=[self.k1*self.a*np.ones(len(X_t)), self.k_1*X_t, self.k2*self.b*np.ones(len(X_t)), self.k_2*Y_t, self.k3*X_t**2*Y_t/self.v**2, self.k_3*X_t**3/self.v**2]
        Entropy_pt_d=np.zeros(len(t));
        for j in range(3):
            Entropy_pt_d[:]=Entropy_pt_d[:]+(current_d[2*j][:]-current_d[2*j+1][:])*np.log(current_d[2*j][:]/current_d[2*j+1][:])    
        self.dermin_entropy=np.mean(Entropy_pt_d)   
        peakind_scipy = signal.argrelextrema(Y_t, comparator=np.greater,order=2)[0]
        if len(peakind_scipy)>6:
            self.dermin_period = t[peakind_scipy[6]]-t[peakind_scipy[5]];
        self.par_ini = np.array([int(X_t[-1]),int(Y_t[-1]),  self.v, self.step])
        self.win_threshold1, self.win_threshold2 = int(np.amax(X_t)*threslold_x), int(np.mean(Y_t)); 
        t0=int(plot_range*self.step*self.Duration*30)
        if plot:
           fig, (ax1,ax2)= plt.subplots(1,2)
           ax1.plot(t[0:t0]*self.t_scale/self.step, X_t[0:t0], label='x(t)')
           ax1.plot(t[0:t0]*self.t_scale/self.step, Y_t[0:t0], label='y(t)')
           ax1.set_xlabel('time')
           ax1.set_ylabel('concentration')
           ax1.legend()
           ax2.scatter(X_t, Y_t, label='trace',s=1)
           ax2.scatter(self.win_threshold1, self.win_threshold2, s =200, marker ="*", c='coral' ,zorder=100 )
           ax2.set_xlabel('x')
           ax2.set_ylabel('y') 
           ax2.legend()
           return self.dermin_entropy, self.dermin_period,fig
        else:
           return self.dermin_entropy, self.dermin_period, 'None'

########################################################################
    def Brusse_Barato(self, Y, t,par):
        [a, b, k1, k2, k3, k_1, k_2, k_3]=par
        v=self.v
        x, y = Y
        dYdt = np.array([a*k1-k_1*x+k3*(x**2)*y/v**2-k_3*(x**3)/v**2, k2*b-k_2*y+k_3*(x**3)/v**2-k3*(x**2)*y/v**2])
        return dYdt
########################################################################
    def Stochastic_simulation(self,):
        Brusse_Barato_loopProgress(self.filename1,self.filename2,self.filename3, self.numberofrealisations,self.win_threshold1, self.win_threshold2, self.par, self.par_ini)
        return 0

    def Window_analysis(self, ):
        df2=pd.read_csv(self.filename2, sep=',')
        T_fpt, Counter_fpt = df2[df2.keys()[0]].values, df2[df2.keys()[1]].values
        counter_his=df2[df2.keys()[2]].values
        counter_his = counter_his[~np.isnan(counter_his)]
        Threshold_FPT1=self.dermin_period*0.5
        Threshold_FPT2=self.dermin_period*1.8
        counter_his=counter_his[np.where(counter_his>Threshold_FPT1)]
        counter_his=counter_his[np.where(counter_his<Threshold_FPT2)]
        self.window_period=np.mean(counter_his)
        self.varT=np.var(counter_his)
        return  self.window_period,self.varT    

    def Autocorrelation_analysis(self,):
        df1=pd.read_csv(self.filename1, sep=',')
        array=df1[' M']
        def AutoCorrelation(x):
            x = np.asarray(x)
            y = x-x.mean()
            result = np.correlate(y, y, mode='full')
            result = result[len(result)//2:]
            result /= result[0]
            return result 
        aucol=AutoCorrelation(array)
        start=0
        duration=int(self.Duration/10);
        # Fit the first set
        fitfunc = lambda p, x: np.sin(2*np.pi/p[0]*x+p[2])*np.exp(-x/p[1]) # Target function
        errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
        p0 = [10., 1.,1.0] # Initial guess for the parameters
        X, Y=np.linspace(0,duration,duration)*1.0/self.step, aucol[start:start+duration]
        p1, success = optimize.leastsq(errfunc, p0[:], args=(X, Y))
        self.autocorrelation_period=np.abs(p1[0])
        self.tauc=p1[1]
        self.D=self.autocorrelation_period**2/self.tauc
        return  self.autocorrelation_period,  self.tauc       
         
