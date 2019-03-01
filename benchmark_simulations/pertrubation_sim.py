import warnings
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.metrics import f1_score
from simulations import core_noise,cos_stable,sin_decay,\
                        sigmoid,biphasic_sigmoid,impulse
from simulations import temporal_sim
from gemelli.tensor_factorization import TenAls
from gemelli.tensor_preprocessing import tensor_rclr
warnings.filterwarnings("ignore")

fold = 10
res = {}
# for each fold
for fold_n in range(fold):
    print('Fold: '+str(fold_n+1))
    for sig_ in [0.05, 0.075, 
                 0.15, 0.2, 0.4, 
                 0.6, 0.8, 0.9]:
        print('    SNR: '+str(sig_))
        # SNR
        noise_ = np.around(1-sig_,2)
        # choose sig. randomly
        sig_nr = np.around(sig_/3,2)
        # perams
        g1_perams = {'perc_stable':sig_,'perc_unstable':0,
                      'perc_impulse':0,'perc_sigmoid_growth':0,
                      'perc_biphas_growth':0,'perc_sigmoid_decay':0,
                      'perc_biphas_decay':0,'perc_noisy':noise_}
        g2_perams = {'perc_stable':0,'perc_unstable':0,
                      'perc_impulse':sig_nr,
                      'perc_sigmoid_growth':sig_nr,
                      'perc_biphas_growth':0,
                      'perc_sigmoid_decay':sig_nr,
                      'perc_biphas_decay':0,'perc_noisy':noise_}
        group_perams_ = {0:g1_perams,1:g2_perams}
        # num features, samp ; time start, end, step, t_peturb
        x,y,time = temporal_sim(500,20,0,3,100,
                                65,group_perams_)
        #truth labels for pertubation in flux
        truth = [0]*60 + [1]*(80-60) + [0]*(100-80)
        # save truth labels 
        if sig_==0 and fold_n==0:
            np.savetxt('pertubation_detection/truth.tsv', 
                       truth, delimiter='\t')
        # run tf
        T_rclr = tensor_rclr(y.copy())
        TF = TenAls(rank=5).fit(T_rclr)
        # save time loadings
        np.savetxt('pertubation_detection/fold_'+str(fold_n)+\
                   '_SNR_'+str(sig_)+'_time_loadings.tsv', 
                   TF.time_loading, delimiter='\t')
        # run change point analysis in grid search
        accuracys = []
        accuracy_labels = []
        # diff jumps
        for j in range(1,11):
            algo = rpt.Pelt(model="rbf",
                            jump=j).fit(TF.time_loading)
            # diff pens 
            for i in range(0,11):
                result = algo.predict(pen=i)
                # generate labels 
                test = [0]*result[0] + [y for i in range(1,len(result)) 
                                        for y in [i%2]*(result[i]-result[i-1])]
                accuracy_labels.append(test)
                accuracys.append(f1_score(truth,test))
        # save highest accruacy from exhaustive grid search
        np.savetxt('pertubation_detection/fold_'+str(fold_n)+\
                   '_SNR_'+str(sig_)+'_prediction.tsv', 
                   accuracy_labels[accuracys.index(max(accuracys))], 
                   delimiter='\t')        
        # get accuracy 
        res[(fold_n,sig_)] = [max(accuracys)]
# save results 
resdf = pd.DataFrame(res).T
resdf.index.names = ['Fold','SNR']
resdf.columns = ['Accuracy']
resdf.to_csv('pertubation_bench_res.tsv',sep='\t')
    