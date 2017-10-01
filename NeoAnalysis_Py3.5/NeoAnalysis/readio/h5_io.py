import h5py as hp
import numpy as np
import re
def h5_io(filename, spike_to_load, analog_to_load):
    spikes = dict()
    analogs = dict()
    events = dict()
    comments = dict()
    with hp.File(filename,'r') as f:
        for key in f.keys():
            if key=='events':
                events['times'] = f[key]['times'].value
                events['labels'] = f[key]['labels'].value
            elif key=='comments':
                comments['times'] = f[key]['times'].value
                comments['labels'] = f[key]['labels'].value
            elif key=='spikes':
                for tem_key in f[key].keys():
                    if tem_key in spike_to_load:
                        spikes[tem_key] = f[key][tem_key]['times'].value
            elif key=='analogs':
                for tem_key in f[key].keys():
                    if tem_key in analog_to_load:
                        analogs[tem_key] = dict()
                        analogs[tem_key]['data'] = f[key][tem_key]['data'].value
                        analogs[tem_key]['sampling_rate'] = f[key][tem_key]['sampling_rate'].value
                        analogs[tem_key]['start_time'] = f[key][tem_key]['start_time'].value
    return events,comments,spikes,analogs