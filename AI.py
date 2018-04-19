from AI_lib import *
from multiprocessing import Pool
import itertools
import pandas as pd
from datetime import datetime
showtime =datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
file_name='Data/AI_Clock'+showtime+'.csv'
v_list=[10,20,40,60,80,100,200]
gamma_list=np.logspace(-3, -2.8, num=20)
#v_list=np.logspace(1.0, 2.0, num=2)
#gamma_list=np.logspace(-8, -3, num=4)
columns=['volume','gamma','entropy_production','period_determin','period_window','period_autocorrelation', 'variance_period','tauc_correlation','D']
para_df = pd.DataFrame(columns=columns)
para_array= itertools.product(v_list,gamma_list)
def func_parallel(par):
    v=par[0];
    gamma=par[1]
    Model=AI_simulation(par)
    Model.determinstic_simulation()
    Model.dermin_entropy
    Model.dermin_period
    Model.Stochastic_simulation()
    Model.Window_analysis()
    Model.Autocorrelation_analysis()
    return pd.DataFrame([[v,gamma, Model.dermin_entropy, Model.dermin_period, Model.window_period, Model.autocorrelation_period, Model.varT, Model.tauc, Model.D ]],columns=columns)

pool = Pool(processes = 27)
results = pool.map(func_parallel, para_array)
pool.close()
pool.join()
results_df = pd.concat(results)
results_df.to_csv(file_name)      
