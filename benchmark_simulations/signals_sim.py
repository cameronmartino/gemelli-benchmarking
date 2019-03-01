import pickle
import numpy as np
import pandas as pd
from simulations import core_noise,cos_stable,sin_decay,\
                        sigmoid,biphasic_sigmoid,impulse
from simulations import temporal_sim
from gemelli.tensor_factorization import TenAls
from gemelli.tensor_preprocessing import tensor_rclr

""" 
A simulation of common signal types across SNR. 
Benchmarking of signal extraction ability. 
"""

# default (will chnage for each type)
g1_perams = {'perc_stable':0,'perc_unstable':0,
              'perc_impulse':0,'perc_sigmoid_growth':0,
              'perc_biphas_growth':0,'perc_sigmoid_decay':0,
              'perc_biphas_decay':0,'perc_noisy':0}
g2_perams = {'perc_stable':0,'perc_unstable':0,
              'perc_impulse':0,'perc_sigmoid_growth':0,
              'perc_biphas_growth':0,'perc_sigmoid_decay':0,
              'perc_biphas_decay':0,'perc_noisy':0}
# oscilations test sig/noise ten-fold
res = {}
n_folds = 10
# test signals for each type
steps_ = 50
time = np.linspace(0,1,steps_)
timeimp = np.linspace(0,.5,steps_)
logittime = np.linspace(-5,5,steps_)
biphastime = np.linspace(-5,5,steps_//2)
sig_tsts = [cos_stable(time),sin_decay(time),impulse(timeimp),
            sigmoid(logittime),biphasic_sigmoid(biphastime),
            sigmoid(logittime)[::-1],
            biphasic_sigmoid(biphastime)[::-1]]
# signal ident. in perams dict
sig_types = ['perc_stable','perc_unstable','perc_impulse',
             'perc_sigmoid_growth','perc_biphas_growth',
             'perc_sigmoid_decay','perc_biphas_decay']
# real signal names 
sig_names = ['Stable Oscillations','Underdamped Oscillations',
             'Impulse','Sigmoid Growth','Biphasic Sigmoid Growth',
             'Sigmoid Sucession','Biphasic Sigmoid Sucession']
# simulation scales
times = [(0,1,steps_),(0,1,steps_),(0,.5,steps_),
         (-5,5,steps_),(-5,5,steps_),
         (-5,5,steps_),(-5,5,steps_)]
# run for n folds
for fold_ in range(n_folds):
    print('Fold: '+str(fold_+1))
    res[fold_] = {}
    # run for all signals
    for sig_tst,sig_type,sig_name,t_typ in zip(sig_tsts,sig_types,sig_names,times):
        print('    '+str(sig_name))
        for sig_ in np.around(np.linspace(0,1,5,endpoint=False),1):
            # noise perc
            if sig_ not in res[fold_].keys():
                res[fold_][sig_] = {}
            noise_ = np.around(1-sig_,1)
            # percent signals
            g1_perams_tmp = g1_perams.copy()
            g1_perams_tmp[sig_type] = sig_
            g1_perams_tmp['perc_noisy'] = noise_
            g2_perams_tmp = g2_perams.copy()
            g2_perams_tmp[sig_type] = sig_
            g2_perams_tmp['perc_noisy'] = noise_    
            group_perams_ = {0:g1_perams_tmp,1:g2_perams_tmp}
            # num features, samp ; time start, end, step, t_peturb
            x,y,time = temporal_sim(500,20,t_typ[0],t_typ[1],t_typ[2],
                                    None,group_perams_)
            # save to pkl
            out_n = 'signals_intermediate/fold_'+str(fold_+1)+'_sim_'\
                    +str(sig_name)+'_SNR_'+str(sig_)+'.pkl'
            output = open(out_n, 'wb')
            pickle.dump(y, output)
            output.close()
            # CLR take reduction rank=1
            T_rclr = tensor_rclr(y.copy())
            TF = TenAls(rank=1).fit(T_rclr)
            res[fold_][sig_]['CLR-'+sig_name] = np.corrcoef(TF.time_loading.ravel(), 
                                                            sig_tst)[1,0]**2
            # no CLR (y needs to be transposed) rank=1
            TF = TenAls(rank=1).fit(y.T)
            res[fold_][sig_]['Raw Counts-'+sig_name] = np.corrcoef(TF.time_loading.ravel(), 
                                                                   sig_tst)[1,0]**2
# write out results 
resdf = pd.DataFrame({(k+1, k2): v2 for k,v in res.items() for k2,v2 in v.items()}).T
resdf.index.names = ['Fold','SNR']
resdf.to_csv('signal_bench_res.tsv',sep='\t')
