import numpy as np
import scipy

from d3m.container import ndarray as d3m_ndarray
from d3m.container import List

def preemphasis(x, coef=0.97):
    return x - np.c_[x[..., :1], x[..., :-1]] * coef

def power_bit_length(x):
    return 2**(x-1).bit_length()

def mel_fbank_filters(nfft, sampfreq, n_chans=20,
                    l_lreq=0.0, h_freq=None):
    if not h_freq: h_freq = 0.5 * sampfreq
    nfft2 = nfft//2+1

    fbin_mel = mel(np.arange(nfft2, dtype=float) * sampfreq / nfft)
    cbin_mel = np.linspace(mel(l_lreq), mel(h_freq), n_chans + 2)
    cind = np.floor(inv_mel(cbin_mel) / sampfreq * nfft).astype(int) + 1
    filters = np.zeros((nfft2, n_chans))
    for i in range(n_chans):
        filters[cind[i]  :cind[i+1], i] = (cbin_mel[i]  -fbin_mel[cind[i]  :cind[i+1]]) / (cbin_mel[i]  -cbin_mel[i+1])
        filters[cind[i+1]:cind[i+2], i] = (cbin_mel[i+2]-fbin_mel[cind[i+1]:cind[i+2]]) / (cbin_mel[i+2]-cbin_mel[i+1])
    if l_lreq > 0.0 and float(l_lreq)/sampfreq*nfft+0.5 > cind[0]: filters[cind[0],:] = 0.0 # Just to be HTK compatible
    return filters

def inv_mel(x):
    return (np.exp(x/1127.)-1.)*700.

def mel(x):
    return 1127.*np.log(1. + x/700.)

#class FixedLenSegmentation(d3m_ndarray):
#    pass
#
#class VarLenSegmentation(List[d3m_ndarray]):
#    pass

