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

def Circle_loopProgress(char* fname1, fname2, int numberOfReactions, win_threshold1, win_threshold2, np.ndarray[DTYPE2_t, ndim=1] reactionrate, par_ini):
    cdef double MT, KT, S, a0,  a1, a2, d1, d2, k0, k1, k2,k3, k4, f1, f2, f_1, f_2, R, X, M, MR, Mp, MpK, K, time, rr[2], step, v,error
    cdef int N, counter, i, j,n_c, h
    cdef double action_rate[8], sum_rate[8], time0, current[8], entropy_sum, entropy,e_s
    
    error=1.0
    R, X, MR, Mp, MpK,M, v, step = par_ini
    MT, KT, S, a1, a2, d1, d2, k0, k1, k2,k3, k4, f1, f2, f_1, f_2 = reactionrate
   # print R, X, MR, Mp, MpK,M
    File1 = open(fname1,"w")
    File1.write('time, R, X, M, MR, Mp, MpK, counter, entropy\n')
    File2 = open(fname2,"w")
    File2.write('time, counter, duration \n')
    entropy=0;
    K = KT - MpK;
    M = MT - Mp -MR- MpK;
    i=0; time=0.00; counter=0; time0=0
    for i in xrange(numberOfReactions):
        e_s=0;
        entropy_sum=0;
        while time < i/step:
              M0=M;
              action_rate =[a1*M*(R)/v, a2*Mp*K/v, d1*MR, d2*MpK,f1*MR, f_1*Mp*(R)/v,f_2*M*K/v, f2*MpK]
              error=1
              current=[f2*MpK,f_2*M*K/v, a2*Mp*K/v, d2*MpK,  f1*MR, f_1*Mp*(R)/v,a1*M*(R)/v, d1*MR]
              entropy=0;
              for h in xrange(4):
                  entropy=entropy+(current[2*h]-current[2*h+1])*log(current[2*h]/current[2*h+1]) 
              if entropy!=float('NaN') and entropy!=float('Inf'):
                 entropy_sum=entropy_sum+entropy
                 e_s=e_s+1      
              a0=action_rate[0]+action_rate[1]+action_rate[2]+action_rate[3]+action_rate[4]+action_rate[5]+action_rate[6]+action_rate[7]  
              #print 'sum of rates', sum_rate
              rr[0], rr[1]=(rand()+1.00)/(RAND_MAX + 1.00), (rand()+1.00)/(RAND_MAX + 1.00)   
              time=time+(1.000/a0)*log(1.000/rr[0]);
              if rr[1]*a0<action_rate[0]:   # a1*M*(R-MR)/v
                 MR = MR+1;  
                 M = M-1;
              elif rr[1]*a0<action_rate[0]+action_rate[1]:   # a2*Mp*K/v
                 Mp = Mp-1;   
                 MpK = MpK+1;     
                 K=K-1;
              elif rr[1]*a0<action_rate[0]+action_rate[1]+action_rate[2]:# d1*MR
                 MR = MR -1;   
                 M=M+1;
              elif rr[1]*a0<action_rate[0]+action_rate[1]+action_rate[2]+action_rate[3]:   #d2*MpK
                 MpK = MpK -1; 
                 Mp = Mp + 1;
                 K=K+1;
              elif rr[1]*a0<action_rate[0]+action_rate[1]+action_rate[2]+action_rate[3]+action_rate[4]: #f1*MR
                 MR = MR -1;
                 Mp = Mp+1;
              elif rr[1]*a0<action_rate[0]+action_rate[1]+action_rate[2]+action_rate[3]+action_rate[4]+action_rate[5]: # f_1*Mp*(R-MR)/v
                 MR = MR +1;
                 Mp = Mp-1;
              elif rr[1]*a0<action_rate[0]+action_rate[1]+action_rate[2]+action_rate[3]+action_rate[4]+action_rate[5]+action_rate[6]:  #f_2*M*K/v  
                 M = M-1;
                 MpK=MpK+1;
                 K=K-1;
              elif rr[1]*a0<action_rate[0]+action_rate[1]+action_rate[2]+action_rate[3]+action_rate[4]+action_rate[5]+action_rate[6]+action_rate[7]: #f2*MpK
                 MpK=MpK-1;
                 M=M+1;
                 K=K+1;
              M = MT - Mp - MR - MpK;
              K = KT - MpK;

              if M0 == win_threshold1 and MR < win_threshold2 and M==win_threshold1+1:
                  counter=counter+1;
                  File2.write(str(time)+', '+str(counter)+','+ str(time-time0)+'\n')
                  time0=time
              if M0 == win_threshold1+1 and MR < win_threshold2 and M==win_threshold1:
                  counter=counter-1;  
                  File2.write(str(time)+', '+str(counter)+'\n')   
        #current=[f2*MpK,f_2*M*K/v+0.1, a2*Mp*K/v+0.1, d2*MpK,  f1*MR,  f_1*Mp*(R-MR)/v,a1*M*(R - MR)/v ,d1*MR] 
        #entropy_p=0;
        #print R, X, M, MR, Mp, MpK
        #for n_c in xrange(4):         
        #    entropy_p= entropy_p+(current[2*n_c]-current[2*n_c+1])*np.log(current[2*n_c]/current[2*n_c+1]);

        output1 = str(time)+','+str(R)+','+str(X)+','+str(M)+','+str(MR)+','+str(Mp)+','+ str(MpK) +','+str(counter) +','+str(entropy_sum/e_s)+'\n';
        File1.write(output1)
    File1.close()    
    File2.close()
    return 0