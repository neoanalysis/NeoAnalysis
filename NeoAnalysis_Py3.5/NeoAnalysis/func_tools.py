from scipy import signal
import numpy as np
import os
try:
    from scipy.signal import sosfilt
    from scipy.signal import zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos

from matplotlib import mlab
from matplotlib.colors import Normalize
import numpy as np
import math as M
import matplotlib.pyplot as plt

def is_num(ite_val):
    try:
        float(ite_val)
    except ValueError:
        return False
    return True

def band_pass(ite_data,freqmin,freqmax,samp_freq,corners=32):
    fe = samp_freq/2.0
    low = freqmin/fe
    high = freqmax/fe
    # raise error for illegal input
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or above Nyquist ({}). Applying a high-pass instead.").format(freqmax, fe)
        return False
    if low >1 :
        msg = "selected low corner requency is above Nyquist"
        return False
    
    z,p,k = signal.iirfilter(corners,[low,high],btype='band',ftype='butter',output='zpk')
    sos = zpk2sos(z,p,k)
    ite_data = sosfilt(sos,ite_data)
    ite_data = sosfilt(sos,ite_data[::-1])[::-1]
    return ite_data

def band_stop(ite_data,freqmin,freqmax,samp_freq,corners=4):
    fe = samp_freq/2.0
    low = freqmin/fe
    high = freqmax/fe
    # raise error for illegal input
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or above Nyquist ({}). Applying a high-pass instead.").format(freqmax, fe)
        return False
    if low >1 :
        msg = "selected low corner requency is above Nyquist"
        return False
    
    z,p,k = signal.iirfilter(corners,[low,high],btype='bandstop',ftype='butter',output='zpk')
    sos = zpk2sos(z,p,k)
    ite_data = sosfilt(sos,ite_data)
    ite_data = sosfilt(sos,ite_data[::-1])[::-1]
    return ite_data
def one_elem_list(ite):
    if len(ite)==1:
        return ite[0]
    else:
        return None

def f_filter_limit(data,limit):
    if limit is not False:
        data.query(limit,inplace=True)
        data.index = np.arange(data.shape[0])
    return data
def f_filter_nan(data,filter_nan):
    if filter_nan is not False:
        if isinstance(filter_nan,str):
            data = data[data[filter_nan].notnull()]
        elif isinstance(filter_nan,list):
            for key in filter_nan:
                data = data[data[key].notnull()]
        data.index = np.arange(data.shape[0])
    return data


def _nearest_pow_2(x):
    """
    Find power of two nearest to x
    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0
    :type x: float
    :param x: Number
    :rtype: Int
    :return: Nearest power of 2 to x
    """
    a = M.pow(2, M.ceil(np.log2(x)))
    b = M.pow(2, M.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b
        
def spectrogram1(data, samp_rate, per_lap=0.9, wlen=None, log=False,
                fmt=None, dbscale=False,y_lim=False,title=False,fig_mark=[0],xtime_offset=0,
                mult=8.0, axes=False,zorder=None,
                clip=[0.0, 1.0]):
    samp_rate = float(samp_rate)
    if not wlen:
        wlen = samp_rate/100.0
    npts=len(data)
    nfft = int(_nearest_pow_2(wlen * samp_rate))
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))
    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))
    # data = data - data.mean()
    end = npts / samp_rate
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]
    vmin, vmax = clip
    if vmin < 0 or vmax > 1 or vmin >= vmax:
        msg = "Invalid parameters for clip option."
        raise ValueError(msg)
    _range = float(specgram.max() - specgram.min())
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)
    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0
    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        time = time+xtime_offset
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        ax.pcolormesh(time, freq, specgram, norm=norm,cmap=plt.cm.jet)
    else:
        # this method is much much faster!
        specgram = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time + xtime_offset, time[-1] + halfbin_time +xtime_offset,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        ax.imshow(specgram, interpolation="nearest", extent=extent,cmap=plt.cm.jet)
    if y_lim:
        ax.vlines(fig_mark,y_lim[0],y_lim[1],color='r')
    else:
        ax.vlines(fig_mark,y_lim[0],y_lim[1],color='r')
        
    ax.axis('tight')
    # ax.set_xlim(0,end)
    # if xtime_offset:
    #     ax.set_xlim(0+xtime_offset, end+xtime_offset)
    #     ih = 0
    #     while 0.5*ih<end:
    #         ih = ih+1
    #     xraw_ticks = np.linspace(0,ih*0.5,ih,endpoint=False)
    #     xnew_ticks = xraw_ticks + xtime_offset
    #     ax.set_xticks(list(xraw_ticks),list(xnew_ticks))
    # else:
    #     ax.set_xlim(0, end)

    if y_lim:
        ax.set_ylim(y_lim[0],y_lim[1])
    ax.grid(False)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Frequency [Hz]')
    if title:
        ax.set_title(title)
    return specgram, freq, time

def flatten_dict(d):
    def expand(key, value):
        if isinstance(value, dict):
            return [ (str(key) + '/' + str(k), v) for k, v in flatten_dict(value).items() ]
        else:
            return [ (str(key), value) ]
    items = [ item for k, v in d.items() for item in expand(k, v) ]
    return dict(items)

def list_files(file_path,key_words):
    target_files=[]
    for parent, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            just_arr = []
            for ite in key_words:                
                if ite in filename:
                    just_arr.append(True)
                if len(just_arr) == len(key_words):
                    target_files.append(file_path+filename)
    return target_files

