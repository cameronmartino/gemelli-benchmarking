import numpy as np
np.seterr(divide = 'ignore') 
from scipy.stats import norm
from numpy.random import poisson, lognormal, normal, randint
from skbio.stats.composition import closure, clr

def core_noise(time):
    """ core/noise """
    return abs(np.random.normal(size=len(time)))

def cos_stable(time, amp=.1, f0=50, phase=0):
    """ table oscilation """
    scale = amp*2
    c = (np.cos((amp * np.pi * f0 * time) + phase) + amp)
    c+=abs(np.min(c))
    return c

def sin_decay(time, amp=2, f0=4, tdecay=.5, phase=0):
    """ decaying oscilation """
    dc = (np.exp(-time/tdecay) *\
            np.sin((amp * np.pi * f0 * time)\
                   + phase))[::-1]
    dc+=abs(np.min(dc))
    return dc

def sigmoid(x, min_=0, max_=1):
    """ sigmoid growth/decay """
    s = (1 / (1 + np.exp(-x)))
    return 1*(s-min_)/(max_-min_)

def biphasic_sigmoid(x, min_=0, max_=1):
    """ biphasic sigmoid """
    bs = np.array(list(sigmoid(x,max_=2))\
                  +list(sigmoid(x,max_=2)+.5))
    return bs

def impulse(T, scale=4, cut_low=True):   
    """ impulse response """
    # Compute the impulse response using the sinc function
    h = (1/T)*np.sinc((1/T)*(np.arange(-(len(T)-1)/2, 
                                       (len(T)-1)/2 + 1)))
    # fill nans
    h[np.isnan(h)] = 0 
    # push up from zerv b o
    h+=abs(np.min(h))
    h*=scale
    return h

def temporal_sim(nf,ns,t_min,t_max,t_step,p_t,group_perams,depth=2500,kappa=0.1,scale_noise=False):
    """ Output tnsors of shape (time,features,samples) """
    # number of groups
    n_blocks = len(group_perams.keys())
    ns_g = ns//n_blocks

    # get t till p_t and t after 
    time = np.linspace(t_min,t_max,t_step)
    biphase_time = np.linspace(t_min,t_max,t_step//2)
    if p_t is not None:
        p_t = p_t//1
        time_b4_p_t = time[:p_t]
        time_aft_p_t = time[p_t:]
        # scale for logistic
        siga4 = np.interp(time_aft_p_t, 
                         (time_aft_p_t.min(), 
                         time_aft_p_t.max()), 
                         (-5, +5))
        # scale decay 
        scale_ = 5
        # biphase (not working)
        biphase_time_b4_p_t = biphase_time[:p_t//2]
        biphase_time_aft_p_t = biphase_time[p_t//2:]
        biphase_time_aft_p_t = np.interp(biphase_time_aft_p_t, 
                                        (biphase_time_aft_p_t.min(), 
                                        biphase_time_aft_p_t.max()), (-5, +5))
    else:
        time_b4_p_t = []
        biphase_time_b4_p_t = []
        time_aft_p_t = time
        biphase_time_aft_p_t = biphase_time
        unstable_b4 = []
        scale_ = 1

    # generate time series per group 
    x = []
    for key,perams in group_perams.items():
        # get number of features by signal 
        perc_stable = int(perams['perc_stable']*nf)
        perc_unstable = int(perams['perc_unstable']*nf)
        perc_impulse = int(perams['perc_impulse']*nf)
        perc_sigmoid_g = int(perams['perc_sigmoid_growth']*nf)
        perc_biphas_g = int(perams['perc_biphas_growth']*nf)
        perc_sigmoid_d = int(perams['perc_sigmoid_decay']*nf)
        perc_biphas_d = int(perams['perc_biphas_decay']*nf)
        perc_noisy = int(perams['perc_noisy']*nf)
        # keep cosistent by addding to noisy
        perc_noisy += nf-sum([perc_stable,perc_unstable,
                              perc_impulse,perc_sigmoid_g,
                              perc_biphas_g,perc_sigmoid_d,
                              perc_biphas_d,perc_noisy])
        # n-samples # 
        ns_all_signals = []
        for s in range(ns_g):

            ## generate those signals ## 
            all_signals = []

            # stable osc.
            for i in range(perc_stable):
                all_signals.append(cos_stable(time))
            # noise (scale from small to loud)
            for i in range(perc_noisy):
                if scale_noise==False:
                    all_signals.append(core_noise(time))
                else:
                    all_signals.append(core_noise(time)*(scale_*(i/perc_noisy)))
            # osc unstable 
            for i in range(perc_unstable):
                all_signals.append(np.array([0.75]*len(time_b4_p_t) + list(sin_decay(time_aft_p_t)*scale_)))
            # impulse
            for i in range(perc_impulse):
                all_signals.append(np.array([0.4]*len(time_b4_p_t) + list(impulse(time_aft_p_t))))
            # sigmoid growth
            for i in range(perc_sigmoid_g):
                all_signals.append(np.array([0]*len(time_b4_p_t) + list(sigmoid(siga4))))
            # biphas sigmoid growth
            for i in range(perc_biphas_g):
                all_signals.append(np.array([0]*len(biphase_time_b4_p_t) + list(biphasic_sigmoid(biphase_time_aft_p_t))))
            # sigmoid decay
            for i in range(perc_sigmoid_d):
                all_signals.append(np.array([.99]*len(time_b4_p_t) + list(sigmoid(siga4))[::-1]))
            # biphas sigmoid decay
            for i in range(perc_biphas_d):
                all_signals.append(np.array([.99]*len(biphase_time_b4_p_t) + list(biphasic_sigmoid(biphase_time_aft_p_t))[::-1]))
            # save samples
            ns_all_signals.append(np.array(all_signals))
        x.append(np.array(ns_all_signals))

    # base-truth
    x = np.vstack(x)
    # sub-sample
    muse = [depth*closure(x[:,:,i]).T for i in range(x.shape[-1])]
    y = np.stack([np.vstack([poisson(lognormal(np.log(muse[i][:, j]), kappa)) 
                              for j in range(muse[i].shape[1])]).T
                   for i in range(x.shape[-1])]).T
    x, y = x.T,y.T
    x[x<0] = 0
    y[y<0] = 0
    return x, y, time

def normal_noise(nf,ns,hodepth,kappa):
    """ uniform-lognormal-poisson normally dist. noise """
    x_noise = abs(normal(1,0.2,(nf,ns)))
    mu = hodepth * closure(x_noise.T).T
    y_noise = np.vstack([poisson(lognormal(np.log(mu[:, i]), 
                                           kappa)) 
                         for i in range(ns)]).T
    return y_noise

def random_noise(nf,ns,hedepth,kappa):
    """ random uniform-lognormal-poisson normally dist. noise """
    x_noise = abs(normal(1,0.2,(nf,ns)))
    err = np.ones_like(x_noise)
    i = randint(0, err.shape[0], 5000)
    j = randint(0, err.shape[1], 5000)
    err[i, j] = hedepth
    x_noise = abs(normal(x_noise, err))
    mu = hedepth * closure(x_noise.T).T
    y_noise = np.vstack([poisson(lognormal(np.log(mu[:, i]), kappa)) 
                              for i in range(ns)]).T
    return y_noise

def chain(gradient, mu, sigma):
    """ gradient normally distributed over species """
    xs = [norm.pdf(gradient, 
                   loc=mu[i], 
                   scale=sigma[i]) 
          for i in range(len(mu))]
    return np.vstack(xs)
    
def gradient(nf, ns, kappa=0.1, depth=100, 
                      sigma=2.0, g_min=0, gmax=10):
    """ poisson-lognormal simulation """
    sigma = [sigma] * nf
    g = np.linspace(g_min, gmax, ns)
    mu = np.linspace(0, 10, nf)
    x = chain(g, mu=mu, sigma=sigma)
    mu = depth * closure(x.T).T
    y = np.vstack([poisson(lognormal(np.log(mu[:, i]), kappa)) 
                   for i in range(ns)]).T
    return x, y

def blocks(nf,ns,kappa=0.1, depth=1500, sigma=5.0, g_min=5, gmax=5,
           sig_noise=.8, noise_ratio=.6, n_blocks=2,
           feature_offset=0, sample_offset=0):
    """ block-simulation """

    # set block index
    ns_index = [i*(ns//n_blocks) for i in range(n_blocks+1)]
    nf_index = [i*(nf//n_blocks) for i in range(n_blocks+1)]
    x = np.zeros((nf,ns))
    y = np.zeros((nf,ns))

    # depths
    noise_depth = int(depth*(1-sig_noise))
    signal_depth = depth - noise_depth
    hodepth = int(noise_depth*noise_ratio)
    hedepth = int(noise_depth*(1-noise_ratio))

    # generate/subsample homoscedastic noise
    y_homo_noise = normal_noise(nf,ns,hodepth,kappa)

    # generate/subsample heteroscedastic noise
    y_hetero_noise = random_noise(nf,ns,hedepth,kappa)

    # generate blocks tables
    for i in range(1,n_blocks+1):

        rev_fea = nf_index[i-1]-((i-1)*feature_offset)
        fwd_fea = nf_index[i]+((i)*feature_offset)

        if fwd_fea>nf:
            fwd_fea=nf

        rev_samp = ns_index[i-1]-((i-1)*sample_offset)
        fwd_samp = ns_index[i]+((i)*sample_offset)

        if fwd_samp>ns:
            fwd_samp=ns

        x_block, y_block = gradient(fwd_samp-rev_samp, fwd_fea-rev_fea, 
                                    depth=signal_depth, sigma=sigma,  
                                    g_min=g_min, gmax=gmax)

        x[rev_fea:fwd_fea,rev_samp:fwd_samp] += x_block.T

        y[rev_fea:fwd_fea,rev_samp:fwd_samp] += y_block.T

    # check subsample overlap
    mu = (signal_depth) * closure(y.T).T
    y = np.vstack([poisson(lognormal(np.log(mu[:, i]), kappa)) 
                   for i in range(ns)]).T

    # add noise
    y+=y_homo_noise
    y+=y_hetero_noise

    # return 
    x = x[:nf_index[-1],:ns_index[-1]]
    y = y[:nf_index[-1],:ns_index[-1]]
          
    return x,y 
