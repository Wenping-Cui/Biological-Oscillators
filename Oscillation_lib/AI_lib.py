from scipy.integrate import odeint
from scipy import optimize
from scipy import signal
import time
import seaborn
from Activator_inhibitor_cython import loopProgress
from scipy.integrate import odeint
import pylab as plt
import numpy as np
import time
import imp
import pandas as pd
#pdb.set_trace()
##########################################################################


class AI_simulation(object):
    def __init__(self, par):
        self.v=par[0]
        self.gamma = par[1];
        self.MT, self.KT, self.S= 10.0*self.v, 1.0*self.v, 0.4*self.v;
        self.k0, self.k1, self.k2, self.k3, self.k4 =  1.0, 1.0, 1.0, 1.0, 0.5;
        self.a1, self.a2 = 100.0, 100.0;
        self.f1, self.f2 = 15.0, 15.0;
        self.d1, self.d2 = 15.0, 15.0;
        self.f_1, self.f_2 = np.sqrt(self.gamma)*self.a1*self.f1/self.d1, np.sqrt(self.gamma)*self.a1*self.f1/self.d1
########################################################################## 
# determinstic values      
        self.entropy=0;
        self.period=0;
        self.R, self.X, self.MR, self.Mp, self.MpK=10,1,1,1,1
        self.Y0 = [self.R, self.X, self.MR, self.Mp, self.MpK]
        self.step=10.0
        self.Duration =20000.0
        self.filename1 = 'Data/AI_data_'+'Data_gamma_'+str(self.gamma)+'_size_'+str(self.v)+'_.csv'
        self.filename2 = 'Data/AI_data_'+'FPT_'+'gamma_'+str(self.gamma)+'_size_'+str(self.v)+'_.csv'
        self.filename3 = 'Data/AI_data_'+'FPTDirect_'+'gamma_'+str(self.gamma)+'_size_'+str(self.v)+'_.csv'
        self.numberofrealisations=int(self.Duration*self.step);
        self.dermin_period=0
        self.dermin_entropy=0;
        self.window_period=0
        self.D=0
        self.tauc=0
        self.varT=0
        self.autocorrelation_period=0
        self.win_threshold1, self.win_threshold2=0,0;
        self.par = np.array([self.MT, self.KT, self.S, self.a1, self.a2, self.d1, self.d2, self.k0, self.k1, self.k2,self.k3, self.k4, self.f1, self.f2, self.f_1, self.f_2])
    def determinstic_simulation(self,):
        t = np.linspace(0, self.Duration/2, 20*self.Duration/2)
        sol = odeint(self.Activator_inhibitor_determin, self.Y0, t, args=(self.par,))
        M_sol = np.ones(len(t))*self.MT-sol[:, 2]-sol[:, 3]-sol[:, 4]
        Rt_d, Xt_d, MRt_d, Mpt_d, MpKt_d, Mt_d  = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4],M_sol[:];
        Kt_d=self.KT*np.ones(len(t))-sol[:, 4];
        current_d=[self.f2*MpKt_d,self.f_2*Mt_d*Kt_d/self.v, self.a2*Mpt_d*Kt_d/self.v, self.d2*MpKt_d,  self.f1*MRt_d, self.f_1*Mpt_d*(Rt_d-MRt_d)/self.v,self.a1*Mt_d*(Rt_d - MRt_d)/self.v, self.d1*MRt_d]
        Entropy_pt_d=np.zeros(len(t));
        for j in range(4):
            Entropy_pt_d[:]=Entropy_pt_d[:]+(current_d[2*j][:]-current_d[2*j+1][:])*np.log(current_d[2*j][:]/current_d[2*j+1][:])
        self.dermin_entropy=np.mean(Entropy_pt_d)   
        peakind_scipy = signal.argrelextrema(M_sol, comparator=np.greater,order=2)[0]
        self.dermin_period = t[peakind_scipy[6]]-t[peakind_scipy[5]];
        self.par_ini = np.array([int(Rt_d[-1]),int(Xt_d[-1]), int(MRt_d[-1]), int(Mpt_d[-1]), int(MpKt_d[-1]), int(Mt_d[-1]),  self.v, self.step])
        self.win_threshold1, self.win_threshold2 = int(np.mean(M_sol)), int(np.mean(sol[:, 2])); 
        return self.dermin_entropy, self.dermin_period


    def Activator_inhibitor_determin(self,Yt, t,par):
        [MT, KT, S, a1, a2, d1, d2, k0, k1, k2,k3, k4, f1, f2, f_1, f_2] =par
        R, X, MR, Mp, MpK  = Yt
        M = MT - Mp -MR- MpK;
        K = KT - MpK; 
        dRdt = k0*Mp + k1*S - k2*X*R/self.v;
        dXdt = k3*Mp - k4*X;
      #  dMdt = f2*MpK + d1*MR - a1*M*(R - MR)-f_2*M*K;
        dMRdt = a1*M*(R - MR)/self.v+f_1*Mp*(R-MR)/self.v - (f1+d1)*MR;
        dMpdt = f1*MR +d2*MpK -a2*Mp*K/self.v- f_1*Mp*(R-MR)/self.v;
        dMpKdt = a2*Mp*K/self.v +f_2*M*K/self.v -f2*MpK-d2*MpK
        dYdt = np.array([dRdt, dXdt,dMRdt,dMpdt,dMpKdt])
        return dYdt


    def Stochastic_simulation(self,):
        loopProgress(self.filename1,self.filename2,self.filename3, self.numberofrealisations,self.win_threshold1, self.win_threshold2, self.par, self.par_ini)
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

    def FPT_analysis(self,):
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
    def Fano_analysis(self,):
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
      
         
    
         
