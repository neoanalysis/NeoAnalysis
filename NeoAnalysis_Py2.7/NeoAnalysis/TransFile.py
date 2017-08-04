# -*- coding: utf-8 -*-
'''
Module for converting recording files from different data acquiring systems to HDF5 format. 
Due to lack of testing data, only data from Blackrock and Plexon were well tested. Data from other acquiring systems 
were not tested. If you have these demo files, we are glad to help you test these data.

This work is based on:
    * Bo Zhang, Ji Dai - first version
'''
from .neo import io
import h5py as hp
import numpy as np
import quantities as pq
from . import func_tools as ftl

class transfile():
    '''
    The class for converting different recording files to HDF5 format in a well-defined data structure.
    This class uses the open-source package Neo as data I/O from different data acquiring systems.
    Data structure in the converted HDF5 file:
        spike_channelNumber_unitNumber: contain spike timestamps and waveforms
        analog_channelNumber: contain data values, sampling rate and the start recording time
        neo_events: contain data values and timestamps, which is equal to events in Neo. Further functions, generate_comments and generate_events
                    need to be used to extract comments and events information from neo_events
    Args
        filename (string):
            File name (without extension)
        machine_type (string):
            Which data acquiring system is used for recording the file, like 'blackrock'. 
            We are glad to provide other file interface for users. Please contact us through email bozhang23@outlook.com
        replace (Boolean):
            If true, data already existed in the converted file will be overwrited.
        **arg (Dict):
            Extra arguments needed for specific machine_type.
            For 'blackrock':
                nsx_to_load, restriction of .nsx files is needed to be translated.
                channels, restriction of spikes channels is needed to be translated.
                units, restriction of spikes units is needed to be translated.
    Returns
        -
    Examples
        >>> TransFile('myfile','blackrock',True,nsx_to_load=[2,4])
            Translate 'myfile', recorded by 'blackrock' machine, to HDF5 format. Both .ns2
            and .ns4 file will be translated (.nev will always be translated).
    '''
    def __init__(self,filename,machine_type,replace=False,**arg):
        """
        Initialize the TransFile class.
        """
        self.filename = filename
        self.replace = replace
        if machine_type == 'blackrock':
            self.__transBlackrock(**arg)
        if machine_type == 'plexon':
            self.__transPlexon(**arg)
        if machine_type == 'alphaomega':
            self.__transAlphaomega(**arg)

    def __transAlphaomega(self):
        r = io.AlphaOmegaIO(filename = self.filename+'.mpx')
        bl = r.read_block(lazy=False,cascade=True)
        seg = bl.segments[0]
        
    def __transBlackrock(self,nsx_to_load='none',channels='all',units='all'):
        '''
        Translate files recorded by the Blackrock data acquiring system.
        '''
        r = io.BlackrockIO(self.filename)
        bl = r.read_block(nsx_to_load=nsx_to_load, load_events=True, load_waveforms=True, channels=channels, units=units)
        seg = bl.segments[0]
        
        # spikes entity
        spikes = dict()
        spikes['spikes'] = dict()
        for spk in seg.spiketrains:
            chan = spk.annotations['ch_idx']
            unit = spk.annotations['unit_id']
            time = np.array(spk.times.rescale(pq.ms))
            waveform = np.array(spk.waveforms.rescale(pq.mV))
            waveform.shape = waveform.shape[0],waveform.shape[-1]
            key = 'spike_'+str(chan)+'_'+str(unit)
            spikes['spikes'][key] = {'times':time,'waveforms':waveform}
            
        # analogs entity
        analogs = dict()
        analogs['analogs'] = dict()
        for ang in seg.analogsignals:
            if ang.shape[0]>0:
                key = 'analog_'+str(ang.annotations['ch_idx'])
                times = np.array(ang.times.rescale(pq.ms))
                data = np.array(ang.rescale(pq.mV))
                data.shape = data.shape[0]
                sampling_rate = int(ang.sampling_rate.rescale(pq.Hz))
                analogs['analogs'][key] = {'data':data, 'start_time':times[0], 'sampling_rate':sampling_rate}
        
        # event entity
        events = dict()
        events['neo_events'] = dict()
        for evt in seg.events:
            key = evt.name
            times = np.array(evt.times.rescale(pq.ms))
            labels = np.array(evt.labels)
            events['neo_events'][key] = {'times':times, 'labels':labels}

        trans_file = self.filename + '.h5'
        self.__save_data(spikes, analogs, events, trans_file)
    
    def __transPlexon(self):
        '''
        Translate files recording by Plexon data acquiring system.
        '''
        r = io.PlexonIO(filename=self.filename+'.plx')
        seg = r.read_segment(lazy=False,cascade=True)

        # spikes entity
        spikes = dict()
        spikes['spikes'] = dict()
        for spk in seg.spiketrains:
            chan = spk.annotations['ch_idx']
            unit = spk.annotations['unit_id']
            time = np.array(spk.times.rescale(pq.ms))
            waveform = np.array(spk.waveforms.rescale(pq.mV))
            waveform.shape = waveform.shape[0],waveform.shape[-1]
            key = 'spike_'+str(chan)+'_'+str(unit)
            spikes['spikes'][key] = {'times':time,'waveforms':waveform}        
        
        # analogs entity
        analogs = dict()
        analogs['analogs'] = dict()
        for ang in seg.analogsignals:
            if ang.shape[0]>0:
                key = 'analog_'+str(ang.annotations['channel_index'])
                times = np.array(ang.times.rescale(pq.ms))
                data = np.array(ang.rescale(pq.mV))
                data.shape = data.shape[0]
                sampling_rate = int(ang.sampling_rate.rescale(pq.Hz))
                analogs['analogs'][key] = {'data':data,'start_time':times[0],'sampling_rate':sampling_rate}

        # event entity
        events = dict()
        events['neo_events'] = dict()
        for evt in seg.events:
            if evt.times.shape[0]>0:
                key = evt.annotations['channel_name']
                times = np.array(evt.times.rescale(pq.ms))
                labels = np.array(evt.labels)
                events['neo_events'][key] = {'times':times, 'labels':labels}

        trans_file = self.filename+'.h5'
        self.__save_data(spikes, analogs, events,trans_file)
    
    def __save_data(self,spikes,analogs,events, trans_file):
        '''
        Save data to HDF5 file in the well-defined data structure.
        '''
        self.trans_file = trans_file
        
        spikes = ftl.flatten_dict(spikes)
        analogs = ftl.flatten_dict(analogs)
        events = ftl.flatten_dict(events)
        
        flat_data = dict()
        flat_data.update(spikes)
        flat_data.update(analogs)
        flat_data.update(events)
        
        keys_in = list()
        if self.replace is False:
            with hp.File(self.trans_file,'a') as f:
                for key in flat_data.keys():
                    if key in f:
                        keys_in.append(key)
        
        if len(keys_in) > 0:
            print(keys_in)
            raise ValueError("These data already in file, please set replace = True to update these data")
        else:
            with hp.File(self.trans_file,'a') as f:
                for ite_k,ite_v in flat_data.iteritems():
                    if ite_k in f:
                        del f[ite_k]
                    f[ite_k] = ite_v
                f.flush()
            print("Data are stored now.")
            
            
def generate_comments(filename, method, replace=False, **arg):
    '''
    method:
        'move':move data from neo_events to comments
        'map': map data in neo_events to comments
    '''
    if method == 'move':
        with hp.File(filename,'a') as f:
            comments_exist = 'comments' in f
            if (replace is False) and comments_exist:
                raise ValueError("comments already exists in the file, please set replace = True to replace it")
            if (replace is True) and comments_exist:
                del f['comments']
            
            # read labels and times of comments
            labels = f['neo_events'][arg['key']]['labels'].value
            times = f['neo_events'][arg['key']]['times'].value
            
            # write labels and times to comments
            f['comments/labels'] = labels
            f['comments/times'] = times

    elif method == 'map':
        with hp.File(filename,'a') as f:
            comments_exist = 'comments' in f
            if (replace is False) and comments_exist:
                raise ValueError("comments already exists in the file, please set replace = True to replace it")
            if (replace is True) and comments_exist:
                del f['comments']
            
            if len(arg['mapping'])>0:
                comments_times = []
                comments_labels = []
                for map_key, to_value in arg['mapping'].iteritems():
                    if isinstance(to_value,str):
                        evt_key, evt_value = map_key.split('/')
                        mask = f['neo_events'][evt_key]['labels'].value == evt_value
                        evt_times = f['neo_events'][evt_key]['times'].value[mask]
                        evt_labels = np.full(evt_times.shape,to_value,dtype='|S%s'%(len(to_value)))
                        comments_times.append(evt_times)
                        comments_labels.append(evt_labels)
                    elif isinstance(to_value,list):
                        evt_key, evt_value = map_key.split('/')
                        mask = f['neo_events'][evt_key]['labels'].value == evt_value
                        evt_times = f['neo_events'][evt_key]['times'].value[mask]
                        for temp_value in to_value:
                            evt_labels = np.full(evt_times.shape,temp_value,dtype='|S%s'%(len(temp_value)))
                            comments_times.append(evt_times)
                            comments_labels.append(evt_labels)
                        
                comments_times = np.concatenate(comments_times,axis=0)
                comments_labels = np.concatenate(comments_labels,axis=0)
                sorted_index = np.argsort(comments_times)
                f['comments/labels'] = comments_labels[sorted_index]
                f['comments/times'] = comments_times[sorted_index]

def generate_events(filename, method, replace=False, **arg):
    '''
    method:
        'move':move data from events to events
        'map': map data in events to events
    '''
    if method == 'move':
        with hp.File(filename,'a') as f:
            events_exist = 'events' in f
            if (replace is False) and events_exist:
                raise ValueError("events already exists in the file, please set replace = True to replace it")
            if (replace is True) and events_exist:
                del f['events']
            
            # read labels and times of comments
            labels = f['neo_events'][arg['key']]['labels'].value
            times = f['neo_events'][arg['key']]['times'].value
            
            # write labels and times to comments
            f['events/labels'] = labels
            f['events/times'] = times
    elif method == 'map':
        with hp.File(filename,'a') as f:
            events_exist = 'events' in f
            if (replace is False) and events_exist:
                raise ValueError("events already exists in the file, please set replace = True to replace it")
            if (replace is True) and events_exist:
                del f['events']
            
            if len(arg['mapping'])>0:
                events_times = []
                events_labels = []
                for map_key,to_value in arg['mapping'].iteritems():
                    evt_key,evt_value = map_key.split('/')
                    mask = f['neo_events'][evt_key]['labels'].value == evt_value
                    evt_times = f['neo_events'][evt_key]['times'].value[mask]
                    evt_labels = np.full(evt_times.shape,to_value,dtype='|S%s'%(len(to_value)))
                    events_times.append(evt_times)
                    events_labels.append(evt_labels)
                events_times = np.concatenate(events_times,axis=0)
                events_labels = np.concatenate(events_labels,axis=0)
                sorted_index = np.argsort(events_times)
                f['events/labels'] = events_labels[sorted_index]
                f['events/times'] = events_times[sorted_index]
                
                
            
                