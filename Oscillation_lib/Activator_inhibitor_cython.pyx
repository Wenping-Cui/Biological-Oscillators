# module for the loops in Gillespie algorithm
# python cython_setup.py build_ext --inplace
import numpy as np
cimport numpy as np
import time
cimport cython
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, fseek, ftell, SEEK_END, rewind, fread
from numpy cimport float64_t


cdef extern from "math.h":
    double log(float theta)
    double sin(double x)
    int abs (int x)
    
    
cdef extern from "stdlib.h":
    double rand()
    void srand(unsigned int seed )
    double RAND_MAX
    int round(double x)    
    
from libc.stdio cimport *
from libc.limits cimport *

DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE2 = np.float
ctypedef np.float_t DTYPE2_t
@cython.cdivision(True)

def loopProgress(char* fname1, fname2, fname3, int numberOfReactions, win_threshold1, win_threshold2, np.ndarray[DTYPE2_t, ndim=1] reactionrate, par_ini):
    cdef double MT, KT, S, a0,  a1, a2, d1, d2, k0, k1, k2,k3, k4, f1, f2, f_1, f_2, R, X, M, MR, Mp, MpK, K, time, rr[2], step, v,time1
    cdef int N, counter, counter1, i, j,n_c, h, error, particle
    cdef double action_rate[13], sum_rate[13], time0, current[8], entropy_sum, entropy,e_s,M0, MR0
    
    error=1
    R, X, MR, Mp, MpK,M, v, step = par_ini
    MT, KT, S, a1, a2, d1, d2, k0, k1, k2,k3, k4, f1, f2, f_1, f_2 = reactionrate
   # print R, X, MR, Mp, MpK,M
    File1 = open(fname1,"w")
    File1.write('time, R, X, M, MR, Mp, MpK, counter, entropy\n')
    File2 = open(fname2,"w")
    File2.write('time, counter, duration \n')
    File3 = open(fname3,"w")
    File3.write('time, counter, duration \n')
    filename3=''
    entropy=0;
    particle=0;
    K = KT - MpK;
    M = MT - Mp -MR- MpK;
    action_rate =[k0*Mp, k1*S, k2*X*R/v,k3*Mp, k4*X,a1*M*(R-MR)/v, a2*Mp*K/v, d1*MR, d2*MpK,f1*MR, f_1*Mp*(R-MR)/v,f_2*M*K/v, f2*MpK]
    j=0; sum_rate[0]=0;sum_rate[12]=0;
    while j<len(action_rate):
        sum_rate[j] = sum_rate[j-1]+action_rate[j]
        j=j+1;   
    i=0; time=0.00; counter=0; time0=0; time1=0; counter1=0;
    for i in xrange(numberOfReactions):
        e_s=0;
        entropy_sum=0;
        while time < i/step:
              M0=M;
              MR0=MR;
              action_rate =[k0*Mp, k1*S, k2*X*R/v,\
                               k3*Mp, k4*X,\
                               a1*M*(R-MR)/v, a2*Mp*K/v, d1*MR, d2*MpK,\
                               f1*MR, f_1*Mp*(R-MR)/v,\
                             f_2*M*K/v, f2*MpK]
              error=1;
              current=[f2*MpK,f_2*M*K/v, a2*Mp*K/v, d2*MpK,  f1*MR, f_1*Mp*(R-MR)/v,a1*M*(R - MR)/v, d1*MR]
              for h in xrange(len(action_rate)):
                  if action_rate[h]<0 and (h==5 or h==10):
                     #print 'error',j
                     error =-1;  
                     action_rate[h]=error*action_rate[h]
                  sum_rate[j]=0;                      
              if current[5]<0:      
                  current[5]=-current[5]
                  current[6]=-current[6]
              entropy=0;
              for h in xrange(4):
                  entropy=entropy+(current[2*h]-current[2*h+1])*log(current[2*h]/current[2*h+1]) 
              if entropy!=float('NaN') and entropy!=float('Inf'):
                 entropy_sum=entropy_sum+entropy
                 e_s=e_s+1      
             # print action_rate 
              for h in xrange(len(action_rate)):
                    sum_rate[h] = sum_rate[h-1]+action_rate[h]
              a0=sum_rate[12]  
              #print 'sum of rates', sum_rate
              rr[0], rr[1]=(rand()+1.00)/(RAND_MAX + 1.00), (rand()+1.00)/(RAND_MAX + 1.00)   
              time=time+(1.000/a0)*log(1.000/rr[0]);
              if rr[1]*a0<sum_rate[0]:#k0*Mp
                 R = R +1;
              elif rr[1]*a0<sum_rate[1]: # k1*S
                 R = R +1;
              elif rr[1]*a0<sum_rate[2]: #k2*X*R/v
                 R = R -1;     
              elif rr[1]*a0<sum_rate[3]:   #k3*Mp
                 X = X+1;
              elif rr[1]*a0<sum_rate[4]:  #  k4*X
                 X = X-1;
################################################################################################################
              elif rr[1]*a0<sum_rate[4]+ error*a1*(R-MR)/v and divmod(particle,4)[1]==0:   # particle transition from M to MR
                 MR = MR+error*1;   
                 M = M-error*1;
                 particle=particle-error*1;   
              elif rr[1]*a0<sum_rate[5]:   # a1*M*(R-MR)/v
                 MR = MR+error*1;   
                 M = M-error*1;
################################################################################################################
              elif rr[1]*a0<sum_rate[5]+ a2*K/v and (divmod(particle,4)[1]==2):   # particle transition from Mp to MpK
                 Mp = Mp-1;   
                 MpK = MpK+1;     
                 K=K-1;
                 particle=particle-1;      
              elif rr[1]*a0<sum_rate[6]:   # a2*Mp*K/v
                 Mp = Mp-1;   
                 MpK = MpK+1;     
                 K=K-1;
################################################################################################################
              elif rr[1]*a0<sum_rate[6]+ d1 and (divmod(particle,4)[1]==3):   # particle transition from MR to M
                 MR = MR -1;   
                 M=M+1;
                 particle=particle+1;         
              elif rr[1]*a0<sum_rate[7]:# d1*MR
                 MR = MR -1;   
                 M=M+1;
################################################################################################################ 
              elif rr[1]*a0<sum_rate[7]+ d2 and (divmod(particle,4)[1]==1):   # particle transition from MpK to Mp
                 MpK = MpK -1; 
                 Mp = Mp + 1;
                 K=K+1;
                 particle=particle+1;                   
              elif rr[1]*a0<sum_rate[8]:   #d2*MpK
                 MpK = MpK -1; 
                 Mp = Mp + 1;
                 K=K+1;
################################################################################################################  
              elif rr[1]*a0<sum_rate[8]+ f1 and (divmod(particle,4)[1]==3):   # particle transition from MR to Mp
                 MR = MR -1;
                 Mp = Mp+1;
                 particle=particle-1;     
              elif rr[1]*a0<sum_rate[9]: #f1*MR
                 MR = MR -1;
                 Mp = Mp+1;
################################################################################################################ 
              elif rr[1]*a0<sum_rate[9]+ error*f_1*(R-MR)/v and (divmod(particle,4)[1]==2):   # particle transition from Mp to MR
                 MR = MR +error*1;
                 Mp = Mp-error*1;
                 particle=particle+error*1; 
              elif rr[1]*a0<sum_rate[10]: # f_1*Mp*(R-MR)/v
                 MR = MR +error*1;
                 Mp = Mp-error*1;
################################################################################################################ 
              elif rr[1]*a0<sum_rate[10]+ f_2*K/v  and (divmod(particle,4)[1]==0):   # particle transition from M to MpK
                 M = M-1;
                 MpK=MpK+1;
                 K=K-1;
                 particle=particle+1; 
              elif rr[1]*a0<sum_rate[11]:  #f_2*M*K/v  
                 M = M-1;
                 MpK=MpK+1;
                 K=K-1;
################################################################################################################ 
              elif rr[1]*a0<sum_rate[11]+ f2 and (divmod(particle,4)[1]==1):   # particle transition from MpK to M
                 MpK=MpK-1;
                 M=M+1;
                 K=K+1;
                 particle=particle-1; 
              elif rr[1]*a0<sum_rate[12]: #f2*MpK
                 MpK=MpK-1;
                 M=M+1;
                 K=K+1;  
################################################################################################################ 
              M = MT - Mp - MR - MpK;
              K = KT - MpK;

              if M0 == win_threshold1 and MR < win_threshold2 and M==win_threshold1+1 and MR0 < win_threshold2:
                  counter=counter+1;
                  File2.write(str(time)+', '+str(counter)+','+ str(time-time0)+'\n')
                  time0=time
              if M0 == win_threshold1+1 and MR < win_threshold2 and M==win_threshold1 and MR0 < win_threshold2 :
                  counter=counter-1;  
                  File2.write(str(time)+', '+str(counter)+'\n')   
              if particle == -4:
                  particle=0;
                  counter1=counter1+1;
                  File3.write(str(time)+', '+str(counter1)+','+ str(time-time1)+','+str(1)+'\n') 
                  time1=time
              elif particle==50:
                  particle=0;
                  counter1=counter1+1;
                  File3.write(str(time)+', '+str(counter1)+','+ str(time-time1)+','+str(-1)+'\n') 
                  time1=time        
        output1 = str(time)+','+str(R)+','+str(X)+','+str(M)+','+str(MR)+','+str(Mp)+','+ str(MpK) +','+str(counter) +','+str(entropy_sum/e_s)+'\n';
        File1.write(output1)
    File1.close()    
    File2.close()
    File3.close()
    return 0