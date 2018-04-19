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

def Brusse_Barato_loopProgress(char* fname1, fname2, fname3, int numberOfReactions, win_threshold1, win_threshold2, np.ndarray[DTYPE2_t, ndim=1] reactionrate, np.ndarray[DTYPE2_t, ndim=1] par_ini):
    cdef double a, b,k1, k2, k3, k_1, k_2, k_3, x, y, time, rr[2], step, v,time1
    cdef int N, counter, counter1, i, j,n_c, h, particle
    cdef double action_rate[6], sum_rate[6], time0, current[6], entropy_sum, entropy,e_s

    x, y, v, step = par_ini
    a, b, k1, k2, k3, k_1, k_2, k_3 = reactionrate
    print k_2
    File1 = open(fname1,"w")
    File1.write('time, x, y, counter, entropy \n')
    File2 = open(fname2,"w")
    File2.write('time, counter, duration \n')
    entropy=0;
    action_rate =[a*k1, k_1*x, k2*b, k_2*y, k3*x**2*y/v**2,k_3*x**3/v**2]
    i=0; time=0.00; counter=0; time0=0; time1=0; counter1=0;
    for i in range(numberOfReactions):
        e_s=0;
        entropy_sum=0;
        while time < i/step:
              x0=x;
              y0=y;
              action_rate =[a*k1, k_1*x, k2*b, k_2*y, k3*x**2*y/v**2,k_3*x**3/v**2]
              current=[a*k1, k_1*x, k2*b, k_2*y, k3*x**2*y/v**2,k_3*x**3/v**2]
              entropy=0;
              for h in range(3):
                  entropy=entropy+(current[2*h]-current[2*h+1])*log(current[2*h]/current[2*h+1]) 
              if entropy!=float('NaN') and entropy!=float('Inf'):
                 entropy_sum=entropy_sum+entropy
                 e_s=e_s+1           
              a0=a*k1+k_1*x+k2*b+k_2*y+k3*x**2*y/v**2+k_3*x**3/v**2
              #print 'sum of rates', sum_rate
              rr[0], rr[1]=(rand()+1.00)/(RAND_MAX + 1.00), (rand()+1.00)/(RAND_MAX + 1.00)   
              time=time+(1.000/a0)*log(1.000/rr[0]);
################################################################################################################
              if rr[1]*a0<a*k1:   # a*k1
                 x=x+1
              elif rr[1]*a0<a*k1+k_1*x:   # k_1*x
                 x=x-1    
              elif rr[1]*a0<a*k1+k_1*x+k2*b:# k2*b
                 y=y+1              
              elif rr[1]*a0<a*k1+k_1*x+k2*b+k_2*y:   #k_2*y
                 y=y-1; 
              elif rr[1]*a0<a*k1+k_1*x+k2*b+k_2*y+k3*x**2*y/v**2: #k3*x**2*y/v**2
                 x=x+1
                 y=y-1
              elif rr[1]*a0<a0: # k_3*x**3/v**2
                 x=x-1
                 y=y+1
################################################################################################################ 
              if x0 == win_threshold1 and y < win_threshold2 and x==win_threshold1+1 and y0 < win_threshold2:
                  counter=counter+1;
                  File2.write(str(time)+', '+str(counter)+','+ str(time-time0)+'\n')
                  time0=time
              if x0 == win_threshold1+1 and y < win_threshold2 and x==win_threshold1 and y0 < win_threshold2:
                  counter=counter-1;  
                  File2.write(str(time)+', '+str(counter)+'\n')    
        output1 = str(time)+','+str(x)+','+str(y)+','+str(counter) +','+str(entropy_sum/e_s)+'\n';
        File1.write(output1)
    File1.close()    
    File2.close()

def Brusselator_loopProgress(char* fname1, fname2, fname3, int numberOfReactions, win_threshold1, win_threshold2, np.ndarray[DTYPE2_t, ndim=1] reactionrate, np.ndarray[DTYPE2_t, ndim=1] par_ini):
    cdef double a, b, c, k1, k2, k3, k_1, k_2, k_3, x, y, time, rr[2], step, v,time1
    cdef int N, counter, counter1, i, j,n_c, h, particle
    cdef double action_rate[6], sum_rate[6], time0, current[6], entropy_sum, entropy,e_s

    x, y, v, step = par_ini
    a, b, c, k1, k2, k3, k_1, k_2, k_3 = reactionrate
   # print R, X, MR, Mp, MpK,M
    File1 = open(fname1,"w")
    File1.write('time, x, y, counter, entropy \n')
    File2 = open(fname2,"w")
    File2.write('time, counter, duration \n')
    entropy=0;
    action_rate =[a*k1, k_1*x, k2*b*x/v, k_2*c*y/v, k3*x**2*y/v**2,k_3*x**3/v**2]
    i=0; time=0.00; counter=0; time0=0; time1=0; counter1=0;
    for i in xrange(numberOfReactions):
        e_s=0;
        entropy_sum=0;
        while time < i/step:
              x0=x;
              y0=y;
              action_rate =[a*k1, k_1*x, k2*b*x/v, k_2*c*y/v, k3*x**2*y/v**2,k_3*x**3/v**2]
              current=[a*k1, k_1*x, k2*b*x/v, k_2*c*y/v, k3*x**2*y/v**2,k_3*x**3/v**2]
              entropy=0;
              for h in range(3):
                  entropy=entropy+(current[2*h]-current[2*h+1])*log(current[2*h]/current[2*h+1]) 
              if entropy!=float('NaN') and entropy!=float('Inf'):
                 entropy_sum=entropy_sum+entropy
                 e_s=e_s+1      
              for h in range(len(action_rate)):
                   sum_rate[h]=0;  
              sum_rate[0]=action_rate[0];     
              for h in range(len(action_rate)-1):
                    sum_rate[h+1] = sum_rate[h]+action_rate[h+1]    
              a0=a*k1+k_1*x+k2*b*x/v+k_2*c*y/v+k3*x**2*y/v**2+k_3*x**3/v**2  
              #print 'sum of rates', sum_rate
              rr[0], rr[1]=(rand()+1.00)/(RAND_MAX + 1.00), (rand()+1.00)/(RAND_MAX + 1.00)   
              time=time+(1.000/a0)*log(1.000/rr[0]);
################################################################################################################
              if rr[1]*a0<a*k1:   # a*k1
                 x=x+1
              elif rr[1]*a0<a*k1+k_1*x:   # k_1*x
                 x=x-1    
              elif rr[1]*a0<a*k1+k_1*x+k2*b*x/v:# k2*b*x/v
                 x=x-1
                 y=y+1              
              elif rr[1]*a0<a*k1+k_1*x+k2*b*x/v+k_2*c*y/v:   #k_2*c*y/v
                 x=x+1;
                 y=y-1; 
              elif rr[1]*a0<a*k1+k_1*x+k2*b*x/v+k_2*c*y/v+k3*x**2*y/v**2: #k3*x**2*y/v**2
                 x=x+1
                 y=y-1
              elif rr[1]*a0<a0: # k_3*x**3/v**2
                 x=x-1
                 y=y+1
################################################################################################################ 
              if x0 == win_threshold1 and y < win_threshold2 and x==win_threshold1+1 and y0 < win_threshold2:
                  counter=counter+1;
                  File2.write(str(time)+', '+str(counter)+','+ str(time-time0)+'\n')
                  time0=time
              if x0 == win_threshold1+1 and y < win_threshold2 and x==win_threshold1 and y0 < win_threshold2:
                  counter=counter-1;  
                  File2.write(str(time)+', '+str(counter)+'\n')    
        output1 = str(time)+','+str(x)+','+str(y)+','+str(counter) +','+str(entropy_sum/e_s)+'\n';
        File1.write(output1)
    File1.close()    
    File2.close()
    return 0