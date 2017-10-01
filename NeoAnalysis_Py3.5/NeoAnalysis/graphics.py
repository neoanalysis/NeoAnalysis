# -*- coding: utf-8 -*-
'''
The module for groupping data into a table on a trial-by-trial basis according to experimental conditions,
and then providing access for users to perform analysis like plotting PSTH and other common application.
This work is based on:
    * Bo Zhang, Ji Dai - first version
'''
import pandas as pd
import numpy as np
import math
from matplotlib import mlab
import matplotlib.pyplot as plt
from .readio.h5_io import h5_io
import copy
import h5py as hp
from scipy import signal, ndimage
import os
from . import func_tools as ftl
from .separate_trials import separate_trials
from sklearn.neighbors import KernelDensity
from scipy import signal, ndimage, fftpack, stats
from scipy.signal import sosfilt
from scipy.signal import zpk2sos
import seaborn
seaborn.set_style(style='white')

class Graphics():
    '''
    The Class for analyzing data according to experimental conditions. It can analyze spike train,local field potential 
    and behavioral data (e.g. saccade, reaction time) using different display methods.
    
    Args
        filename (string):
            file name (with or without extension)
        trial_start_mark (string):
            The event marker representing the start of a trial, which is used to separate the raw
            data into different trials. 
        comment_expr (string):
            This parameter tells the program how experimental condition and parameters are stored in the data.
            For example, a experimental condition, patch direction, is stored in the way 'patch_direction:degree'.
            By setting the comment_expr as "key:value", the program decodes the key as 'patch_direction', and 
            the value for a particular trial is the degree of that trial.
        spike_to_load (string or list):
            Define the spike channels and units.
            If 'all', spikes in all channels and all units will be loaded.
            If 'none', spike data will not be loaded.
            If set to be a string like 'spike_26_1', spike of unit 1 in 26 channel will be loaded.
            If set to be a list like ['spike_26_1','spike_23_2'], spike of unit 1 in 26 channel and spike of unit 2 in 23 channel will be loaded.
            Default: 'all'
        analog_to_load (string or list):
            Define the analog signal channels.
            if 'all', analog signals in all channels will be loaded.
            if 'none', analog signals will not be loaded.
            if set to be a string like 'analog_25', analog signals in channel 25 will be loaded.
            if set to be a list like ['analog_25','analog_20'], analog signals in channel 25 and channel 20 will be loaded.
            Default: 'none'
    Returns
        -
    Examples
        >>> gh = Graphics('myfile.h5','64715','key:value')
            In this example, event marker '64715' is used to separate the raw data into different trials.
            'key:value' is used to extract experimental condition information.
            This command initiates the Graphics class, and groups all data into a table, wherein each row represents a trial,
            and each column represents a specific data, e.g. stimulus onset time, offset time, reaction time, spike,
            LFP, etc. 
            The command gh.data_df displays the data table
    '''
    def __init__(self,filename,trial_start_mark,comment_expr,spike_to_load='all',analog_to_load='all'):
        """
        Initialize the Graphics class.
        """
        if not filename.endswith('.h5'):
            filename = filename+'.h5'
        self.filename = filename
        # find available spike channnel and analog channel in file
        spike_avail = list()
        analog_avail = list()
        with hp.File(filename,'r') as f:
            for key in f.keys():
                if key == 'spikes':
                    for tem_key in f[key].keys():
                        spike_avail.append(tem_key)
                if key == 'analogs':
                    for tem_key in f[key].keys():
                        analog_avail.append(tem_key)
        # load spike data and analog signals from the file
        spike_to_load = self.__spike_to_load(spike_to_load, spike_avail)
        analog_to_load = self.__analog_to_load(analog_to_load, analog_avail)
        events,comments,spikes,analogs = h5_io(filename, spike_to_load, analog_to_load)
        events['labels'] = [ite.decode() for ite in events['labels']]
        events['labels'] = np.array(events['labels'])
        comments['labels'] = [ite.decode() for ite in comments['labels']]
        comments['labels'] = np.array(comments['labels'])
        self.data_df = separate_trials(events,comments,spikes,analogs,trial_start_mark,comment_expr)
        sampling_rate = dict()
        for chn in analogs.keys():
            sampling_rate[chn] = analogs[chn]['sampling_rate']
        if len(sampling_rate)>0:
            self.sampling_rate = sampling_rate
            
    def __spike_to_load(self,spike_to_load, spike_avail):
        if isinstance(spike_to_load,str):
            if spike_to_load == 'none':
                spike_to_load = []
            elif spike_to_load == 'all':
                spike_to_load = spike_avail
            elif spike_to_load in spike_avail:
                spike_to_load = [spike_to_load] 
            elif spike_to_load in [ite.replace('spike_','') for ite in spike_avail]:
                spike_to_load = ['spike_'+spike_to_load]
            else:
                raise ValueError("Invalid specification of spike_to_load")
        elif spike_to_load is None:
            spike_to_load = []

        if spike_to_load:
            for i,ite in enumerate(spike_to_load):
                if not ite.startswith('spike_'):
                    spike_to_load[i] = 'spike_'+ite
                    if spike_to_load[i] not in spike_avail:
                        raise ValueError('%s not avaliable'%spike_to_load[i])
        return spike_to_load

    def __analog_to_load(self,analog_to_load, analog_avail):
        if isinstance(analog_to_load,str):
            if analog_to_load == 'none':
                analog_to_load = []
            elif analog_to_load == 'all':
                analog_to_load = analog_avail
            elif analog_to_load in analog_avail:
                analog_to_load = [analog_to_load]
            elif analog_to_load in [ite.replace('analog_','') for ite in analog_avail]:
                analog_to_load = ['analog_'+analog_to_load]
            else:
                raise ValueError("Invalid specification of analog_to_load")
        elif analog_to_load is None:
            analog_to_load = []

        if analog_to_load:
            for i,ite in enumerate(analog_to_load):
                if not ite.startswith('analog_'):
                    analog_to_load[i] = 'analog_'+ite
                    if analog_to_load[i] not in analog_avail:
                        raise ValueError('%s not avaliable'%analog_to_load[i])
        return analog_to_load

    # sort data table according to certain columns
    def __sort_by(self,data,sort_by):
        data_group = data.groupby(sort_by)
        return data_group.groups

    # calculate firing rate of spikes in certain channel
    def __cal_firing_rate(self,data,channel,bin_size,overlap,pre_time,post_time):
        bins_left = [pre_time]
        while bins_left[-1] < post_time:
            bins_left.append(bins_left[-1]+bin_size-overlap)
        bins_left = np.array(bins_left)
        bins_right = bins_left+bin_size
        bins_mean = (bins_left+bins_right)/2.0
        zero_offset = bins_mean[bins_mean>0][0]
        bins_left = bins_left - zero_offset
        bins_right = bins_right - zero_offset
        bins_mean = bins_mean - zero_offset

        bins_left = bins_left[bins_right<=post_time]
        bins_mean = bins_mean[bins_right<=post_time]
        bins_right = bins_right[bins_right<=post_time]

        bins_mean = bins_mean[bins_left>=pre_time]
        bins_right = bins_right[bins_left>=pre_time]
        bins_left = bins_left[bins_left>=pre_time]
        def cal_fr(ite_spike):
            ite_fr = list()
            for i in range(bins_left.shape[0]):
                ite_fr_i = ite_spike[(ite_spike>=bins_left[i])&(ite_spike<bins_right[i])].shape[0]
                ite_fr.append(ite_fr_i)
            ite_fr = np.array(ite_fr)
            ite_fr = ite_fr*1000.0/bin_size
            return ite_fr
        firing_rate = data[channel].apply(cal_fr)
        return firing_rate, bins_mean
    
    # Group data by experimental conditions and plot PSTH and raster of each condition
    def plot_spike(self,channel,sort_by,align_to,pre_time,post_time,bin_size=30,overlap=0,Mean=3,Sigma=10,limit=False,filter_nan=False,fig_marker=[0],fig_size=[12,7],fig_column=4,fig_pad=0.5,fig_wspace=0.02,fig_hspace=0.15,figure=True):
        '''
        Args
            channel (string): 
                define the spike channel and unit separated by a dash. Example: chanel_unit
            sort_by (list):
                define the conditions used to sort data
            align_to (string):
                event marker used to align each trial's spikes
            pre_time (int):
                Set the time(msec) before the align_to to be covered
            post_time (int):
                Set the Time(msec) after the align_to to be covered
            bin_size (int):
                bin size (msec) used to calculate PSTH
                Default: 30
            overlap (int):
                overlap (msec) between adjacent bins
                Default: 0
            Mean (float):
                mean of the gaussian kernal used to smooth the PSTH
                Default: 3
            Sigma (float):
                sigma of the gaussian kernal used to smooth the PSTH
                Default: 10
            limit (string):
                an expression used to filtering the data by certain conditions.
                Default: False
            filter_nan (list):
                trials with NaN value in the listed columns will be excluded
                Default: False
            fig_marker (list):
                Define the positions of the reference vertical lines by setting some time points in the list.
                Default: [0]
            fig_size (list):
                define the size of the figure
                Default: [12,7]
            fig_column (int):
                Define the number of sub-plots in each row
                Default: 4
            fig_pad (float):
                the space of padding of the figure
                Default: 0.5
            fig_wspace (float):
                the width reserved for blank space between subplots
                Default: 0.02
            fig_hspace (float):
                the height reserved for white space between subplots
                Default: 0.15
            figure (Boolean):
                if True, the figure will be displayed.
                Default: True
        Returns
            {'data':{condition_1:PSTH,
                     condition_2:PSTH,
                      .
                      .
                      .},
             'time':firing rate time}
        Examples
        >>> firingRate = gh.plot_spike(channel = ‘spike_26_1’, sort_by = [‘patch_direction'], align_to = ’dig_64721’, 
                                                   pre_time = -300, post_time = 2000, bin_size = 30, overlap = 10, filter_nan = 
                                                   [‘dig_64721’,’dig_64722'], fig_column = 4, fig_marker = [0])
        '''
        if pre_time>0:
            raise ValueError('pre_time must <= 0')
        if post_time<=0:
            raise ValueError('post_time must >0')
        data = copy.deepcopy(self.data_df)
        # limit, filter_nan
        data = ftl.f_filter_limit(data,limit)
        data = ftl.f_filter_nan(data,filter_nan)
        # align spike in plot channel to align_to
        data[channel] = data[channel] - data[align_to]
        # sort data by certain experimental conditions in sort_by
        data_group = self.__sort_by(data,sort_by)
        firing_rate,fr_time = self.__cal_firing_rate(data,channel,bin_size,overlap,pre_time,post_time)
        firing_rates_mean = dict()
        spikes_select = dict()
        # build the gaussian filter
        fr_buffer = signal.gaussian(Mean,Sigma)
        scatter_max = list()
        for k,v in data_group.items():
            firing_rate_select = firing_rate.loc[v]
            spikes_select[k] = data[channel].loc[v]
            firing_rate_select = firing_rate_select.mean(axis=0)
            firing_rate_select[fr_time>0] = ndimage.filters.convolve1d(firing_rate_select[fr_time>0], fr_buffer/fr_buffer.sum())
            firing_rate_select[fr_time<=0] = ndimage.filters.convolve1d(firing_rate_select[fr_time<=0], fr_buffer/fr_buffer.sum())
            firing_rates_mean[k] = firing_rate_select
            scatter_max.append(len(v))
        fr_max = max([v.max() for k, v in firing_rates_mean.items()])
        fr_max =  (int(fr_max / 10) + 1) * 10
        scatter_max = max(scatter_max)
        scatter_height = fr_max*1.0/scatter_max
        group_keys = list(data_group.keys())
        group_keys = pd.DataFrame(group_keys, columns=sort_by)
        group_keys = group_keys.sort_values(sort_by)
        group_keys.index = range(group_keys.shape[0])
        x_lim = [pre_time,post_time]
        if figure is True:
            if group_keys.shape[1] == 1:
                block_num = 1
                block_in_num = group_keys.shape[0] / block_num
                row_num = int(math.ceil(float(block_in_num) / fig_column))
                row_nums = row_num
                fig_all = plt.figure(figsize=fig_size)
                for i in group_keys.index:
                    current_block = 1
                    block_in_i = i - (current_block - 1) * block_in_num + 1
                    i_pos = (current_block - 1) * row_num * fig_column + block_in_i
                    plt.subplot(row_nums, fig_column, i_pos)
                    cond_i = group_keys.loc[i].values[0]
                    plt.plot(fr_time, firing_rates_mean[cond_i], linewidth=2.5)
                    for i_spike, ite_spike in enumerate(spikes_select[cond_i]):
                        plt.vlines(ite_spike,fr_max+i_spike*scatter_height,fr_max+i_spike*scatter_height+scatter_height*0.618,color='k',linewidth=0.3,alpha=1)
                    plt.vlines(fig_marker, 0, fr_max * 2, color='red')
                    if i_pos%fig_column ==1:
                        plt.yticks([0,fr_max/2,fr_max])
                    elif i_pos%fig_column != 1:
                        plt.yticks([0,fr_max/2,fr_max],[])
                    plt.yticks([0,fr_max/2,fr_max])
                    plt.xlim(x_lim)
                    plt.ylim(0, fr_max*2)
                    plt.title(str(cond_i))
                fig_all.tight_layout(pad=fig_pad)
                fig_all.subplots_adjust(wspace=fig_wspace,hspace=fig_hspace)
            elif group_keys.shape[1] == 2:
                block_num = group_keys[sort_by[0]].unique().shape[0]
                block_in_num = group_keys.shape[0]/block_num
                row_num = int(math.ceil(float(block_in_num) / fig_column))
                row_nums = block_num * row_num
                fig_all = plt.figure(figsize=fig_size)
                for i in group_keys.index:
                    current_block = int(math.floor(i / block_in_num)) + 1
                    block_in_i = i - (current_block - 1) * block_in_num + 1
                    i_pos = (current_block - 1) * row_num * \
                        fig_column + block_in_i
                    plt.subplot(row_nums, fig_column, i_pos)
                    cond_i = tuple(group_keys.loc[i].values)
                    plt.plot(fr_time, firing_rates_mean[cond_i], linewidth=2.5)
                    for i_spike, ite_spike in enumerate(spikes_select[cond_i]):
                        plt.vlines(ite_spike,fr_max+i_spike*scatter_height,fr_max+i_spike*scatter_height+scatter_height*0.618,color='k',linewidth=1)
                    plt.vlines(fig_marker, 0, fr_max * 2, color='red')
                    if i_pos%fig_column ==1:
                        plt.yticks([0,fr_max/2,fr_max])
                    elif i_pos%fig_column != 1:
                        plt.yticks([0,fr_max/2,fr_max],[])
                    # plt.yticks([0,fr_max/2,fr_max])
                    plt.xlim(x_lim)
                    plt.ylim(0, fr_max * 2)
                    plt.title([str(ite) for ite in cond_i])
                fig_all.tight_layout(pad=fig_pad)
                fig_all.subplots_adjust(wspace=fig_wspace,hspace=fig_hspace)
        return {'data':firing_rates_mean,'time':fr_time}

    # Sort data by experimental conditions and plot the spike count during a given period
    def plot_spike_count(self,channel,sort_by,align_to,timebin,limit=False,filter_nan=False,figure=True):
        '''
        Args
            channel (string): 
                define the spike channel and unit separated by a dash. Example: chanel_unit
            sort_by (list):
                experimental conditions used to sort data
            align_to (string):
                event marker used to align each trial's spikes
            timebin (list):
                Define the period for calculating spike counts.
            limit (string):
                an expression used to filter the data by certain conditions.
                Default: False
            filter_nan (list):
                trials with NaN value in the listed columns will be excluded
                Default: False
            figure (Boolean):
                if True, the figure will be displayed.
                Default: True
        Returns
            {condition_1: {'mean':value,
                           'sem':value}
             condition_2: {'mean':value,
                           'sem':value}
             '
             '
             '
            }

        Examples:
        spk_count = gh.plot_spike_count(channel = ’spike_26_1’, sort_by = [‘patch_direction’], align_to = 
                                                             ’dig_64721’, timebin=[0,700]) 
        '''
        data = copy.deepcopy(self.data_df)
        # limit, filter_nan
        data = ftl.f_filter_limit(data,limit)
        data = ftl.f_filter_nan(data,filter_nan)
        # align spike in plot channel to align_to
        data[channel] = data[channel] - data[align_to]
        # sort data by certain experimental conditions in sort_by
        data_group = self.__sort_by(data,sort_by)
        spikes_count = dict()
        spikes_sem = dict()
        for k,v in data_group.items():
            spikes_count[k] = list()
            data_select = data[channel].loc[v]
            for i in data_select.index:
                tem_spike = data_select.loc[i]
                spike_inbin = np.logical_and(tem_spike >= timebin[0], tem_spike <= timebin[1])
                spikes_count[k].append(spike_inbin.sum())
            spikes_sem[k] = np.std(spikes_count[k],ddof=1)/math.sqrt(len(spikes_count[k]))
            spikes_count[k] = np.mean(spikes_count[k])
        ymin = np.min([v.min() for k,v in spikes_count.items() ])
        ymax = np.max([v.max() for k,v in spikes_count.items() ])
        ymin = 0.9*ymin
        ymax = 1.1*ymax
        group_keys = list(data_group.keys())
        group_keys = pd.DataFrame(group_keys, columns=sort_by)
        group_keys = group_keys.sort_values(sort_by)
        group_keys.index = range(group_keys.shape[0])
        if figure is True:
            plt.figure()
            if group_keys.shape[1] == 1:
                xs = list()
                xticks = list()
                ys = list()
                err = list()
                for i,col_1 in enumerate(group_keys[sort_by[0]].unique()):
                    xs.append(i)
                    xticks.append(col_1)
                    ys.append(spikes_count[col_1])
                    err.append(spikes_sem[col_1])
                plt.errorbar(xs,ys,yerr=err)
                plt.xlim([xs[0]-1, xs[-1]+1])
                plt.xticks(xs,xticks)
                plt.ylim([ymin,ymax])
                plt.legend(framealpha=0,labelspacing=0.01)
                plt.xlabel(sort_by[0])
                plt.ylabel('numbers of spikes')
            elif group_keys.shape[1] == 2:
                for col_2 in group_keys[sort_by[1]].unique():
                    xs = list()
                    xticks = list()
                    ys = list()
                    err = list()
                    for i,col_1 in enumerate(group_keys[sort_by[0]].unique()):
                        if (col_1, col_2) in spikes_count.keys():
                            xs.append(i)
                            xticks.append(col_1)
                            ys.append(spikes_count[(col_1, col_2)])
                            err.append(spikes_sem[(col_1, col_2)])
                    plt.errorbar(xs,ys,yerr=err,label=col_2)
                    plt.xlim(xs[0]-1,xs[-1]+1)
                    plt.xticks(xs,xticks)
                    plt.ylim([ymin,ymax])
                    plt.legend(framealpha=0,labelspacing=0.01)
                    plt.xlabel(sort_by[0])
                    plt.ylabel('numbers of spikes')
        return {k:{'mean':spikes_count[k],'sem':spikes_sem[k]} for k,v in data_group.items()}

    # Sort data by experimental conditions and plot scalar data in lineplot (e.g. reaction time) 
    def plot_line(self,target,sort_by,limit=False,filter_nan=False):
        '''
        Args
            target (string):
                the name of the scalar data to be analyzed
            sort_by (list):
                experimental conditions used to sort data
            limit (string):
                an expression used to filter the data by certain conditions.
                Default: False
            filter_nan (list):
                trials with NaN value in the listed columns will be excluded
                Default: False
        Returns
            {condition_1: {'mean':value,
                           'sem':value,
                           'num':value}
             condition_2: {'mean':value,
                           'sem':value,
                           'num':value}
             '
             '
             '
            }
            
        Examples:
        Reaction_time=gh.plot_line('Reaction_time',sort_by=['a','A'],limit='Reaction_time<500')
        '''
        data = copy.deepcopy(self.data_df)
        # limit, filter_nan
        data = ftl.f_filter_limit(data,limit)
        data = ftl.f_filter_nan(data,filter_nan)
        # sort data by certain experimental conditions in sort_by
        data_group = self.__sort_by(data,sort_by)
        target_select = {}
        target_mean = dict()
        target_sem = dict()
        target_num = dict()
        for k, v in data_group.items():
            target_select[k] = data[target].loc[v]
            target_select[k] = pd.to_numeric(target_select[k])
            target_mean[k] = target_select[k].mean(axis=0)
            target_sem[k] = target_select[k].std(axis=0, ddof=1) / np.sqrt(target_select[k].shape[0])
            target_num[k] = target_select[k].shape[0]
            target_select[k] = [target_select[k].mean(axis=0), target_select[k].std(axis=0, ddof=1) / np.sqrt(target_select[k].shape[0]), target_select[k].shape[0]]
        group_keys = list(data_group.keys())
        group_keys = pd.DataFrame(group_keys,columns=sort_by)
        group_keys = group_keys.sort_values(sort_by)
        group_keys.index = range(group_keys.shape[0])

        plt.figure()
        if group_keys.shape[1] == 1:
            xs = list()
            xticks = list()
            ys = list()
            err = list()
            for i,col_1 in enumerate(group_keys[sort_by[0]].unique()):
                xs.append(i)
                xticks.append(col_1)
                ys.append(target_mean[col_1])
                err.append(target_sem[col_1])
            plt.errorbar(xs,ys,yerr=err)
            plt.xlim([xs[0]-1, xs[-1]+1])
            plt.xticks(xs,xticks)
            plt.legend(framealpha=0,labelspacing=0.01)
            plt.xlabel(sort_by[0])
        elif group_keys.shape[1] == 2:
            for col_2 in group_keys[sort_by[1]].unique():
                xs = list()
                xticks = list()
                ys = list()
                err = list()
                for i,col_1 in enumerate(group_keys[sort_by[0]].unique()):
                    if (col_1, col_2) in target_mean.keys():
                        xs.append(i)
                        xticks.append(col_1)
                        ys.append(target_mean[(col_1, col_2)])
                        err.append(target_sem[(col_1, col_2)])
                plt.errorbar(xs,ys,yerr=err,label=col_2)
                plt.xlim(xs[0]-1,xs[-1]+1)
                plt.xticks(xs,xticks)
                plt.legend(framealpha=0,labelspacing=0.01)
                plt.xlabel(sort_by[0])
        return {k:{'mean':target_mean[k],'sem':target_sem[k],'num':target_num[k]} for k,v in data_group.items()}
    def plot_bar(self,target,sort_by,limit=False,filter_nan=False,ci=95,kind='bar'):
        '''
        Args
            target (string):
                the name of the scalar data to be analyzed
            sort_by (list):
                experimental conditions used to sort data
            limit (string):
                an expression used to filter the data by certain conditions.
                Default: False
            filter_nan (list):
                trials with NaN value in the listed columns will be excluded
                Default: False
            ci (float):
                confidence interval
                defaule: 95
            kind (str):
                The kind of plot to draw., link 'bar', 'point'
                Default: 'bar'
        Returns
            {condition_1: {'mean':value,
                           'sem':value,
                           'num':value}
             condition_2: {'mean':value,
                           'sem':value,
                           'num':value}
             '
             '
             '
            }
            
        Examples:
        Reaction_time=gh.plot_line('Reaction_time',sort_by=['a','A'],limit='Reaction_time<500')
        '''
        data = copy.deepcopy(self.data_df)
        # limit, filter_nan
        data = ftl.f_filter_limit(data,limit)
        data = ftl.f_filter_nan(data,filter_nan)
        # sort data by certain experimental conditions in sort_by
        data_group = self.__sort_by(data,sort_by)
        target_select = {}
        target_mean = dict()
        target_sem = dict()
        target_num = dict()
        for k, v in data_group.items():
            target_select[k] = data[target].loc[v]
            target_select[k] = pd.to_numeric(target_select[k])
            target_mean[k] = target_select[k].mean(axis=0)
            target_sem[k] = target_select[k].std(axis=0, ddof=1) / np.sqrt(target_select[k].shape[0])
            target_num[k] = target_select[k].shape[0]
            target_select[k] = [target_select[k].mean(axis=0), target_select[k].std(axis=0, ddof=1) / np.sqrt(target_select[k].shape[0]), target_select[k].shape[0]]
        group_keys = list(data_group.keys())
        group_keys = pd.DataFrame(group_keys,columns=sort_by)
        group_keys = group_keys.sort_values(sort_by)
        group_keys.index = range(group_keys.shape[0])
        if group_keys.shape[1] == 2:
            g = seaborn.factorplot(x=sort_by[0],y=target,hue=sort_by[1],data=data,size=6,kind=kind,palette='muted',ci=ci,
                                    order=sorted(list(set(data[sort_by[0]].values))),hue_order=sorted(list(set(data[sort_by[1]].values))))
            g.set_ylabels('')
        elif group_keys.shape[1] == 1:
            if isinstance(sort_by,list):
                x_id = sort_by[0]
            elif isinstance(sort_by,str):
                x_id = sort_by
            g = seaborn.factorplot(x=x_id,y=target,data=data,size=6,kind=kind,palette='muted',ci=ci,
                                    order=sorted(list(set(data[sort_by[0]].values))))
            g.set_ylabels('')
            
        return {k:{'mean':target_mean[k],'sem':target_sem[k],'num':target_num[k]} for k,v in data_group.items()}
    # convert data type in certain columns to numberic type
    def to_numeric(self,columns):
        '''
        Args
            columns (string or list):
                column names needed to be converted
        Returns
            -
        '''
        if isinstance(columns,str):
            self.data_df[columns] = pd.to_numeric(self.data_df[columns],errors='coerce')
        elif isinstance(columns,list):
            for column in columns:
                self.data_df[column] = pd.to_numeric(self.data_df[column],errors='coerce')

    # rename certain columns
    def rename(self,names_dict):
        '''
        Args
            names_dict (dict):
                {'old_name_1':'new_name_1',
                 'old_name_2':'new_name_2',
                 .
                 .
                 .}
        Returns
            -
        '''
        self.data_df = self.data_df.rename(columns=names_dict)

    # This function performs adding to a given column
    def df_add(self,column,added_info):
        '''
        Args
            column (string):
                the column name to be played with
            added_info (string, int, float or pandas.DataFrame):
                The information to be added to the selected column can be string, int, float, or 
                pandas.DataFrame
        Returns
            -
        '''
        if isinstance(added_info,str):
            self.data_df[column] = self.data_df[column] + self.data_df[added_info]
        elif isinstance(added_info,(int,float)):
            self.data_df[column] = self.data_df[column] + added_info
        elif isinstance(added_info,(pd.Series,pd.DataFrame)):
            self.data_df[column] = self.data_df[column] + added_info

    # This function performs minus to a given column
    def df_minus(self,column,minus_info):
        '''
        Args
            column (string):
                the column name to be played with
            minus_info (string, int, float or pandas.DataFrame):
                information to be subtracted from the selected column
        Returns
            -
        '''
        if isinstance(minus_info,str):
            self.data_df[column] = self.data_df[column] - self.data_df[minus_info]
        elif isinstance(minus_info,(int,float)):
            self.data_df[column] = self.data_df[column] - minus_info
        elif isinstance(added_info,(pd.Series,pd.DataFrame)):
            self.data_df[column] = self.data_df[column] - added_info

    # This function multiplys the selected column with certain factor
    def df_multiply(self,column,multiply_info):
        '''
        Args
            column (string):
                the column name to be played with
            multiply_info (string, int, float or pandas.DataFrame):
                information to be used for multiplying
        Returns
            -
        '''
        if isinstance(multiply_info,str):
            self.data_df[column] = self.data_df[column] * self.data_df[multiply_info]
        elif isinstance(multiply_info,(int,float)):
            self.data_df[column] = self.data_df[column] * multiply_info
        elif isinstance(added_info,(pd.Series,pd.DataFrame)):
            self.data_df[column] = self.data_df[column] * added_info

    # This function divides the selected column by certain factor
    def df_division(self,column,division_info):
        '''
        Args
            column (string):
                the column name to be played with
            division_info (string, int, float or pandas.DataFrame):
                information to be used for dividing
        Returns
            -
        '''
        if isinstance(division_info,str):
            self.data_df[column] = self.data_df[column] / self.data_df[division_info]
        elif isinstance(division_info,(int,float)):
            self.data_df[column] = self.data_df[column] / division_info
        elif isinstance(added_info,(pd.Series,pd.DataFrame)):
            self.data_df[column] = self.data_df[column] / added_info

    # delete certain trials in the data table
    def del_trials(self,trials):
        '''
        Args
            trials (list):
                indexs of trials to be deleted
        Returns
            -
        '''
        self.data_df.drop(trials,axis=0,inplace=True)
        self.data_df.index = np.arange(self.data_df.shape[0])

    # delete certain columns in the data table
    def del_columns(self,columns):
        '''
        Args
            columns (string, list):
                List the column names to be deleted
        Returns
            -
        '''
        self.data_df.drop(columns,axis=1,inplace=True)

    # add certain column to the data table
    def add_column(self,name,add_data):
        '''
        Args
            name (string, list):
                define the name(s) for the newly added column
            add_data (int, float, string, list, pandas.Series, pandas.DataFrame):
                if int, float or string, all rows of this new column will be filled with this value
                if list, pandas.Series or pandas.DataFrame, their dimensions need to be consistent with the data table
        Returns
            -
        '''
        self.data_df[name] = add_data
    
    # Sort data by experimental conditions and plot analog signals (e.g. LFP)
    def plot_analog(self,channel,sort_by,align_to,pre_time,post_time,limit=False,filter_nan=False,normalize=True,fig_marker=[0],fig_size=[12,7],fig_column=4):
        '''
        Args
            channel (string): 
                define the analog channel
            sort_by (list):
                experimental conditions used to sort data
            align_to (string):
                event marker used to align each trial's signals
            pre_time (int):
                Set the time(msec) before the align_to to be covered
            post_time (int):
                Set the time(msec) after the align_to to be covered
            limit (string):
                an expression used to filter the data by certain conditions.
                Default: False
            filter_nan (list):
                trials with NaN value in the listed columns will be excluded
                Default: False
            fig_marker (list):
                Defines the positions of the reference vertical lines by setting some time points in the list.
                Default: [0]
            fig_size (list):
                the size of the figure
                Default: [12,7]
            fig_column (int):
                number of sub-plots in one row
                Default: 4
        Returns
            {'time': analog signal time
             'data': {'condition_1':signal data,
                      'condition_2':signal data,
                       .
                       .
                       .}
            }
        '''
        if pre_time>0:
            raise ValueError('pre_time must <= 0')
        if post_time<=0:
            raise ValueError('post_time must >0')
        data = copy.deepcopy(self.data_df)
        samp_time = 1000.0/self.sampling_rate[channel]
        # limit, filter_nan
        data = ftl.f_filter_limit(data,limit)
        data = ftl.f_filter_nan(data,filter_nan)
        # align signal time in plot channel to align_to
        # sort data by certain experimental conditions in sort_by
        data_group = self.__sort_by(data,sort_by)
        # align analog_time to zero mark
        points_num = int((post_time-pre_time)/samp_time)
        ana_timestamp = np.linspace(pre_time,pre_time+points_num*samp_time,points_num,endpoint=False)
        start_points = data[align_to]/samp_time - abs(pre_time/samp_time)
        pre_time_num = int(abs(pre_time/samp_time))
        start_points = start_points.astype(int)
        ana_select=list()
        if normalize is True:
            for i in data.index:
                start_point = start_points.loc[i]
                end_point = start_point+points_num
                temp_data = data[channel].loc[i][start_point:end_point]
                # each trial, minus average before zero point
                if pre_time_num>0:
                    temp_data = temp_data - temp_data[:pre_time_num].mean()
                ana_select.append(temp_data)
        elif normalize is False:
            for i in data.index:
                start_point = start_points.loc[i]
                end_point = start_point+points_num
                temp_data = data[channel].loc[i][start_point:end_point]
                ana_select.append(temp_data)          

        data[channel] = ana_select
        target_mean = dict()
        for k,v in data_group.items():
            temp_ana = data[channel].loc[v]
            target_mean[k] = temp_ana.mean(axis=0)

        target_max = max([v.max() for _,v in target_mean.items()])
        # target_max = (int(target_max/10.0)+1)*10
        target_min = min([v.min() for _,v in target_mean.items()])
        # target_min = (int(target_min/10.0)-1)*10

        plt.figure(figsize=fig_size)
        for key,value in target_mean.items():
            plt.plot(ana_timestamp,value,label=str(key))
        plt.vlines(fig_marker,target_min,target_max)
        plt.ylim(target_min,target_max)
        plt.xlabel('Time [ms]',fontsize=16)
        plt.ylabel('mV', fontsize=16)
        plt.legend(labelspacing=0)
        return {'time':ana_timestamp,'data':target_mean}

    # Sort data by experimental conditions and plot spectrum of analog signals (e.g. LFP)
    def plot_spectral(self,channel,sort_by,align_to,pre_time,post_time,limit=False,filter_nan=False,x_lim=[1,100],y_lim=False,log=False,window="hann", nfft=None,detrend="constant",scaling="density",fig_size=[12,7]):
        '''
        Args
            channel (string): 
                define the analog channel
            sort_by (list):
                experimental conditions used to sort data
            align_to (string):
                event marker used to align each trial's signals
            pre_time (int):
                Set the time(msec) before the align_to to be covered
            post_time (int):
                Set the time(msec) after the align_to to be covered
            limit (string):
                an expression used to filter the data by certain conditions.
                Default: False
            filter_nan (list):
                trials with NaN value in the listed columns will be excluded
                Default: False
            x_lim (list):
                set limits of x-axis
                Default: [0,100]
            y_lim (list):
                set limits of y-axis
                Default: False
            window (str,tuple or array_like):
                Desired window to use. see parameter window in scipy.signal.periodogram
                Default: "hanning"
            nfft (int):
                length of the FFT used. If None the length of "x" will be used
            detrend (str, function or False, optional):
                Specifies how to detrend `x` prior to computing the spectrum. see parameter detrend in scipy.signal.periodogram.
                Default: "constant"
            scaling ("density","spectrum"):
                if "density": V**2/Hz
                if "spectrum": V**2
                see scaling paramter in scipy.signal.periodogram
            fig_size (list):
                the size of the figure
                Default: [12,7]

        Returns
            {'frequency': frequency
             'data': {'condition_1':signal data,
                       'condition_2':signal data,
                       .
                       .
                       .}
            }
        '''
        if pre_time>0:
            raise ValueError('pre_time must <= 0')
        if post_time<=0:
            raise ValueError('post_time must >0')

        data = copy.deepcopy(self.data_df)
        samp_time = 1000.0/self.sampling_rate[channel]
        fs = 1000.0/samp_time
        # limit, filter_nan
        data = ftl.f_filter_limit(data,limit)
        data = ftl.f_filter_nan(data,filter_nan)
        # sort data by certain experimental conditions in sort_by
        data_group = self.__sort_by(data,sort_by)
        # align analog_time to zero mark
        points_num = int((post_time-pre_time)/samp_time)
        ana_timestamp = np.linspace(pre_time,pre_time+points_num*samp_time,points_num,endpoint=False)
        start_points = data[align_to]/samp_time - abs(pre_time/samp_time)
        start_points = start_points.astype(int)
        ana_select=list()
        for i in data.index:
            start_point = start_points.loc[i]
            end_point = start_point+points_num
            ana_select.append(data[channel].loc[i][start_point:end_point])
        f_spec = lambda ite:signal.periodogram(ite,fs,window=window,nfft=nfft,detrend=detrend,scaling=scaling)
        ana_spec = list(map(f_spec,ana_select))
        freq = ana_spec[0][0]
        f_temp = lambda ite:ite[1]
        ana_spec = list(map(f_temp,ana_spec))
        data[channel] = ana_select
        data[channel+'_spec'] = ana_spec        

        target_mean = dict()
        for k,v in data_group.items():
            temp_ana = data[channel+'_spec'].loc[v]
            target_mean[k] = temp_ana.mean(axis=0)

        mask = (freq>x_lim[0]) & (freq<x_lim[1])
        plt.figure(figsize=fig_size)
        for key,value in target_mean.items():
            if log is True:
                plt.semilogy(freq[mask],value[mask],label=str(key))
            else:
                plt.plot(freq[mask],value[mask],label=str(key))

        plt.xlabel('Frequency [Hz]',fontsize=16)
        plt.ylabel('PSD (mV**2/Hz)',fontsize=16)
        plt.legend(labelspacing=0)
        return {'frequency':freq,'data':target_mean}

    def __nearest_pow_2(self,x):
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
        a = math.pow(2, math.ceil(np.log2(x)))
        b = math.pow(2, math.floor(np.log2(x)))
        if abs(a - x) < abs(b - x):
            return a
        else:
            return b
    
    # calculate spectrogram of signals
    def __spectrogram(self,data,samp_rate,window,per_lap,wlen,mult):        
        samp_rate = float(samp_rate)
        if not wlen:
            wlen = samp_rate/100.0
        
        npts=len(data)
        nfft = int(self.__nearest_pow_2(wlen * samp_rate))
        if nfft > npts:
            nfft = int(self.__nearest_pow_2(npts / 8.0))
        if mult is not None:
            mult = int(self.__nearest_pow_2(mult))
            mult = mult * nfft
        nlap = int(nfft * float(per_lap))
        end = npts / samp_rate

        window = signal.get_window(window,nfft)
        specgram, freq, time = mlab.specgram(data, Fs=samp_rate,window=window,NFFT=nfft,
                                            pad_to=mult, noverlap=nlap)
        return specgram,freq,time

    # Sort data by experimental conditions and plot spectrogram for analog signals (e.g. LFP)
    def plot_spectrogram(self,channel,sort_by,align_to,pre_time,post_time,limit=False,filter_nan=False,y_lim=[0,100],normalize=True,window="hann",per_lap=0.9,wlen=None,mult=8.0,fig_mark=[0],fig_size=[12,7],color_bar=True,fig_column=4):
        '''
        Args
            channel (string): 
                define the analog channel
            sort_by (list):
                experimental conditions used to sort data
            align_to (string):
                event marker used to align each trial' signals
            pre_time (int):
                Set the time(msec) before the align_to to be covered
            post_time (int):
                Set the time(msec) after the align_to to be covered
            limit (string):
                an expression used to filter the data by certain conditions.
                Default: False
            filter_nan (list):
                trials with NaN value in the listed columns will be excluded
                Default: False
            y_lim (list):
                set limits of y-axis
                Default: [0, 100]
            window (str,tuple or array_like):
                Desired window to use. see parameter window in scipy.signal.spectrogram
                Default: "hann"
            per_lap (float):
                percentage of overlap of sliding window, range (0,1)
                Default: 0.9,
            wlen (int, float):
                Window length for fft in seconds. 
                If None, wlen = samp_rate/100.0
                Default: None
            mult: Pad zeros to length mult * wlen, which makes spectrogram smoother.
                Default: 8.0

            fig_mark (list):
                Draw vertical lines at the time points set in the list.
                Default: [0]
            fig_size (list):
                the size of the figure
                Default: [12,7]
            fig_column (int):
                number of sub-plots in one row
                Default: 4
        Returns
            {'frequency':frequency,
             'time':analog signal time,
             'data':{
                'condition_1': spectrogram value,
                'condition_2': spectrogram value,
                .
                .
                .
            }}
        '''
        if pre_time>0:
            raise ValueError('pre_time must <= 0')
        if post_time<=0:
            raise ValueError('post_time must >0')
        data = copy.deepcopy(self.data_df)
        samp_time = 1000.0/self.sampling_rate[channel]
        fs = self.sampling_rate[channel]
        if fig_mark:
            fig_mark = [ite/1000.0 for ite in fig_mark]
        xtime_offset = pre_time/1000.0
        # limit, filter_nan
        data = ftl.f_filter_limit(data,limit)
        data = ftl.f_filter_nan(data,filter_nan)
        # sort data by certain experimental conditions in sort_by
        data_group = self.__sort_by(data,sort_by)
        # align analog_time to align_to marker
        points_num = int((post_time-pre_time)/samp_time)
        ana_timestamp = np.linspace(pre_time,pre_time+points_num*samp_time,points_num,endpoint=False)
        start_points = data[align_to]/samp_time - abs(pre_time/samp_time)
        start_points = start_points.astype(int)
        # pre_time_num = int(abs(pre_time/samp_time))
        ana_select=list()
        for i in data.index:
            start_point = start_points.loc[i]
            end_point = start_point+points_num
            ana_select.append(data[channel].loc[i][start_point:end_point])

        specg_select = list()
        if normalize is True:
            for i,value in enumerate(ana_select):
                i_specg,i_freq,i_time = self.__spectrogram(value,fs,window=window,per_lap=per_lap,wlen=wlen,mult=mult)
                # i_freq,i_time,i_specg = signal.spectrogram(value,fs,window=window,nperseg=nperseg,noverlap=noverlap,detrend=detrend,scaling=scaling)
                temp_time = 0 - pre_time/1000.0
                mask_time = i_time<temp_time
                temp_specg = i_specg[:,mask_time]
                if temp_specg.shape[1]>0:
                    # check if temp_specg is empty
                    temp_specg = temp_specg.mean(axis=1)
                    temp_specg = temp_specg[:,np.newaxis]
                    # normalize spectrogram of each trial
                    i_specg = i_specg - temp_specg                
                specg_select.append(i_specg)
        elif normalize is False:
            for i,value in enumerate(ana_select):
                i_specg,i_freq,i_time = self.__spectrogram(value,fs,window=window,per_lap=per_lap,wlen=wlen,mult=mult)
                # i_freq,i_time,i_specg = signal.spectrogram(value,fs,window=window,nperseg=nperseg,noverlap=noverlap,detrend=detrend,scaling=scaling)
                specg_select.append(i_specg)

        freq = i_freq
        time = i_time
        data[channel+'_spectrogram'] = specg_select

        target_mean = dict()
        for k,v in data_group.items():
            temp_specg = data[channel+'_spectrogram'].loc[v]
            target_mean[k] = temp_specg.mean(axis=0)
        
        if len(y_lim) != 2:
            raise ValueError('y_lim should be a list with two elements')
        vmin = min([value[(y_lim[0]<freq) & (freq<y_lim[1])].min() for _,value in target_mean.items()])
        vmax = max([value[(y_lim[0]<freq) & (freq<y_lim[1])].max() for _,value in target_mean.items()])
        group_keys = np.array(list(data_group.keys()))
        group_keys = pd.DataFrame(group_keys, columns=sort_by)
        group_keys = group_keys.sort_values(sort_by)
        group_keys.index = range(group_keys.shape[0])
        # return_val = dict()
        # return_val['data'] = dict()

        
        if group_keys.shape[1] == 1:
            block_num = 1
            block_in_num = group_keys.shape[0] / block_num
            row_num = int(math.ceil(float(block_in_num) / fig_column))
            row_nums = row_num
            fig_all = plt.figure(figsize=fig_size)
            for i in group_keys.index:
                current_block = 1
                block_in_i = i - (current_block - 1) * block_in_num + 1
                i_pos = (current_block - 1) * row_num * fig_column + block_in_i
                ax = plt.subplot(row_nums, fig_column, i_pos)
                cond_i = group_keys.loc[i].values[0]

                # plt.pcolormesh(time,freq,target_mean[cond_i])

                halfbin_time = (time[1] - time[0]) / 2.0
                halfbin_freq = (freq[1] - freq[0]) / 2.0
                specgram = np.flipud(target_mean[cond_i])
                # center bin
                extent = (time[0] - halfbin_time + xtime_offset, time[-1] + halfbin_time +xtime_offset,
                        freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
                plt.imshow(specgram, interpolation="nearest", extent=extent,cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
                plt.vlines(fig_mark,y_lim[0],y_lim[1],linewidth=0.8)
                # if scale_bar is True:
                #     plt.colorbar(pad=0.01,aspect=50)
                ax.axis('tight')
                plt.ylim(y_lim[0],y_lim[1])
                
                if i_pos%fig_column ==1:
                    pass
                elif i_pos%fig_column != 1:
                    ax.set_yticks([])
                plt.title(str(cond_i))
            fig_all.tight_layout(pad=0.8)
            fig_all.subplots_adjust(wspace=0.05,hspace=0.15,right=0.9)
            if color_bar is True:
                cax = plt.axes([0.91, 0.1, 0.012, 0.8])
                plt.colorbar(cax=cax)
                cax.set_ylabel('PSD (mV**2/Hz)')
            plt.show()

        elif group_keys.shape[1] == 2:
            block_num = group_keys[sort_by[0]].unique().shape[0]
            block_in_num = group_keys.shape[0]/block_num
            row_num = int(math.ceil(float(block_in_num) / fig_column))
            row_nums = block_num * row_num
            fig_all = plt.figure(figsize=fig_size)
            for i in group_keys.index:
                current_block = int(math.floor(i / block_in_num)) + 1
                block_in_i = i - (current_block - 1) * block_in_num + 1
                i_pos = (current_block - 1) * row_num * \
                    fig_column + block_in_i
                ax=plt.subplot(row_nums, fig_column, i_pos)
                cond_i = tuple(group_keys.loc[i].values)

                # plt.pcolormesh(time,freq,target_mean[cond_i])

                halfbin_time = (time[1] - time[0]) / 2.0
                halfbin_freq = (freq[1] - freq[0]) / 2.0
                specgram = np.flipud(target_mean[cond_i])
                # center bin
                extent = (time[0] - halfbin_time + xtime_offset, time[-1] + halfbin_time +xtime_offset,
                        freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
                plt.imshow(specgram, interpolation="nearest", extent=extent,cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
                plt.vlines(fig_mark,y_lim[0],y_lim[1],linewidth=0.8)
                # if scale_bar is True:
                #     plt.colorbar(pad=0.01,aspect=50)
                ax.axis('tight')
                plt.ylim(y_lim[0],y_lim[1])
                
                if i_pos%fig_column ==1:
                    pass
                elif i_pos%fig_column != 1:
                    ax.set_yticks([])
                plt.title(str(cond_i))
            fig_all.tight_layout(pad=0.8)
            fig_all.subplots_adjust(wspace=0.05,hspace=0.15,right=0.9)

            if color_bar is True:
                cax = plt.axes([0.91, 0.1, 0.012, 0.8])
                plt.colorbar(cax=cax)
                cax.set_ylabel('PSD (mV**2/Hz)')
            plt.show()

        return {'frequency':freq,'time':time+xtime_offset,'data':target_mean}
    
    # save analysis results to the workspace for population level analysis 
    def save_data(self,space_name,data,key,replace=False):
        '''
        Args
            space_name (string):
                file path of the work space for storing analysis results
            data (dict):
                analysis results to be stored
            key (string):
                name the stored results
            replace (Boolean):
                if True, stored results will be rewritted if their key has already existed.
        Returns
            -
        '''
        super_key = self.filename.split('/')[-1][:-3]
        flat_data = ftl.flatten_dict(data)
        keys_in = list()
        if replace is False:
            with hp.File(space_name,'a') as f:
                for ite_k,ite_v in flat_data.items():
                    item_key = super_key+'/'+key+'/'+ite_k
                    if item_key in f:
                        keys_in.append(item_key)

        if len(keys_in) > 0:
            print(keys_in)
            raise ValueError("Those data alreadly in file, please set replace=True to update these data")
        else:
            with hp.File(space_name,'a') as f:
                for ite_k,ite_v in flat_data.items():
                    item_key = super_key+'/'+key+'/'+ite_k
                    if item_key in f:
                        del f[item_key]
                    f[item_key] = ite_v
                f.flush()
            print("Data are stored now")

    # filter analog signals
    def __band_filter(self,ite_data,fs,order,lowcut,highcut,zerophase,btype,ftype,rps=None):
        fe = fs/2.0
        low = lowcut/fe
        high = highcut/fe
        if low<0:
            low=0
        if high>1:
            high=1
        if ftype == "cheby1":
            rp = rps
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = rps
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = rps[0]
            rs = rps[1]
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,[low,high],btype=btype,ftype=ftype,output="zpk")
        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data
    def __highpass_filter(self,ite_data,fs,order,lowcut,zerophase,btype,ftype,rps=None):
        fe = fs/2.0
        low = lowcut/fe
        if low<0:
            low=0
        if ftype == "cheby1":
            rp = rps
            z,p,k = signal.iirfilter(order,low,btype=btype,ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = rps
            z,p,k = signal.iirfilter(order,low,btype=btype,ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = rps[0]
            rs = rps[1]
            z,p,k = signal.iirfilter(order,low,btype=btype,ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,low,btype=btype,ftype=ftype,output="zpk")
        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data
    def __lowpass_filter(self,ite_data,fs,order,highcut,zerophase,btype,ftype,rps=None):
        fe = fs/2.0
        high = highcut/fe
        if high>1:
            high=1
        if ftype == "cheby1":
            rp = rps
            z,p,k = signal.iirfilter(order,high,btype=btype,ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = rps
            z,p,k = signal.iirfilter(order,high,btype=btype,ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = rps[0]
            rs = rps[1]
            z,p,k = signal.iirfilter(order,high,btype=btype,ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,high,btype=btype,ftype=ftype,output="zpk")
        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data

    def analog_filter(self,channel,btype,ftype="butter",order=6,zerophase=True,**args):
        '''
        Args
            channel (string):
                define the analog channel
            btype (string): {‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’}
            ftype : str, optional
                        The type of IIR filter to design:
                        Butterworth : ‘butter’
                        Chebyshev I : ‘cheby1’
                        Chebyshev II : ‘cheby2’
                        Cauer/elliptic: ‘ellip’
                        Bessel/Thomson: ‘bessel’
                    Default: "butter"
            order (int): the order of the filter
            zerophase (bool): 
                If True, apply filter once forwards and once backwards.
                This results in twice the filter order but zero phase shift in the resulting filtered trace.
                Default: True
            **args:
                if btype is bandpass or bandstop:
                    if ftype is butter or bessel:
                        highcut, lowcut
                    if ftype is cheby1:
                        highcut, lowcut, rp
                    if ftype is cheby2:
                        highcut, lowcut, rs
                    if ftype is ellip:
                        highcut, lowcut, rp, rs
                if btype is lowpass:
                    if ftype is butter or bessel:
                        highcut
                    if ftype is cheby1:
                        highcut, rp
                    if ftype is cheby2:
                        highcut, rs
                    if ftype is ellip:
                        highcut, rp, rs
                if btype is highpass:
                    if ftype is butter or bessel:
                        lowcut
                    if ftype is cheby1:
                        lowcut, rp
                    if ftype is cheby2:
                        lowcut, rs
                    if ftype is ellip:
                        lowcut, rp, rs
        Returns
            -
        '''

        fs = self.sampling_rate[channel]
        if btype in ["bandpass","bandstop"]:
            lowcut = args["lowcut"]
            highcut = args["highcut"]
            if ftype in ["butter","bessel"]:
                self.data_df[channel] = self.data_df[channel].apply(self.__band_filter,args=(fs,order,lowcut,highcut,zerophase,btype,ftype))
            elif ftype == "cheby1":
                rps = args["rp"]
                self.data_df[channel] = self.data_df[channel].apply(self.__band_filter,args=(fs,order,lowcut,highcut,zerophase,btype,ftype,rps))
            elif ftype == "cheby2":
                rps = args["rs"]
                self.data_df[channel] = self.data_df[channel].apply(self.__band_filter,args=(fs,order,lowcut,highcut,zerophase,btype,ftype,rps))
            elif ftype == "ellip":
                rps = [args["rp"],args["rs"]]
                self.data_df[channel] = self.data_df[channel].apply(self.__band_filter,args=(fs,order,lowcut,highcut,zerophase,btype,ftype,rps))
        elif btype == "highpass":
            lowcut = args["lowcut"]
            if ftype in ["butter","bessel"]:
                self.data_df[channel] = self.data_df[channel].apply(self.__highpass_filter,args=(fs,order,lowcut,zerophase,btype,ftype))
            elif ftype == "cheby1":
                rps = args["rp"]
                self.data_df[channel] = self.data_df[channel].apply(self.__highpass_filter,args=(fs,order,lowcut,zerophase,btype,ftype,rps))
            elif ftype == "cheby2":
                rps = args["rs"]
                self.data_df[channel] = self.data_df[channel].apply(self.__highpass_filter,args=(fs,order,lowcut,zerophase,btype,ftype,rps))
            elif ftype == "ellip":
                rps = [args["rp"],args["rs"]]
                self.data_df[channel] = self.data_df[channel].apply(self.__highpass_filter,args=(fs,order,lowcut,zerophase,btype,ftype,rps))
        elif btype == "lowpass":
            highcut = args["highcut"]
            if ftype in ["butter","bessel"]:
                self.data_df[channel] = self.data_df[channel].apply(self.__lowpass_filter,args=(fs,order,highcut,zerophase,btype,ftype))
            elif ftype == "cheby1":
                rps = args["rp"]
                self.data_df[channel] = self.data_df[channel].apply(self.__lowpass_filter,args=(fs,order,highcut,zerophase,btype,ftype,rps))
            elif ftype == "cheby2":
                rps = args["rs"]
                self.data_df[channel] = self.data_df[channel].apply(self.__lowpass_filter,args=(fs,order,highcut,zerophase,btype,ftype,rps))
            elif ftype == "ellip":
                rps = [args["rp"],args["rs"]]
                self.data_df[channel] = self.data_df[channel].apply(self.__lowpass_filter,args=(fs,order,highcut,zerophase,btype,ftype,rps))    # # filter analog signals

    # smooth eye movement trajectory and realign eye position to a relatively stable period of time, e.g. during fixation.
    def calibrate_eye(self,eye_channel,realign_mark,realign_timebin,eye_medfilt_win=21,eye_gausfilt_sigma=3):
        '''
        Args
            eye_channel (list):
                the first element is the channel name for the horizontal eye position
                the second element is the channel name for the vertial eye position
            realign_mark (string):
                event marker used to align eye positions
            realign_timebin (list):
                a period of time relative to the realign_mark.
                Example: [0,100]
            eye_medfilt_win (int):
                parameter for the median filter to smooth the eye movement trajectory
            eye_gausfilt_sigma (int):
                sigma of the gaussian kernel to smooth the eye movement trajectory
        Return:
            -
        '''
        samp_time = 1000.0/self.sampling_rate[eye_channel[0]]
        # medfilt eye x, y position
        lamb_medfilt = lambda ite:signal.medfilt(ite,eye_medfilt_win)
        self.data_df[eye_channel[0]] = self.data_df[eye_channel[0]].apply(lamb_medfilt)
        self.data_df[eye_channel[1]] = self.data_df[eye_channel[1]].apply(lamb_medfilt)
        # gaussian filt eye x,y position
        lamb_gausfilt = lambda ite:ndimage.filters.gaussian_filter1d(ite,eye_gausfilt_sigma)
        self.data_df[eye_channel[0]] = self.data_df[eye_channel[0]].apply(lamb_gausfilt)
        self.data_df[eye_channel[1]] = self.data_df[eye_channel[1]].apply(lamb_gausfilt)
        # align eye to realign_mark, realign_timebin uses realign_mark as reference
        realign_poinnum = (self.data_df[realign_mark]/samp_time).values
        start_points = realign_poinnum + realign_timebin[0]/samp_time
        points_num = int((realign_timebin[1]-realign_timebin[0])/samp_time)
        for channel in eye_channel:
            align_points = list()
            for idx in self.data_df.index:
                start_point = start_points[idx]
                if ~np.isnan(start_point):
                    start_point = int(start_point)
                    end_point = start_point + points_num
                    align_point = self.data_df[channel].loc[idx][start_point:end_point]
                    align_point = align_point.mean()
                else:
                    align_point = np.nan
                align_points.append(align_point)
            self.data_df[channel] = self.data_df[channel] - align_points

    # find all saccades for all trials
    def find_saccade(self,eye_channel,eye_speed_win=5,sac_speed_threshold=100,sac_duration_threshold=10,sac_displacement_threshold=2):
        '''
        Args
            eye_channel (list):
                the first element is the channel name for the horizontal eye position
                the second element is the channel name for the vertial eye position
            eye_speed_wind (int):
                Number of points to calculate eye movement speed
            sac_speed_threshold (int):
                Set the speed threshold for a valid saccade
                Default: 100
            sac_duration_threshold (int):
                Set the (minimum) duration threshold for a valid saccade.
                Default: 10 (msec)
            sac_displacement_threshold (int):
                Set the minimum saccade amplitude for a valid saccade
                Default: 2
        Returns
            -
        '''
        data = copy.deepcopy(self.data_df[eye_channel])
        samp_time = 1000.0/self.sampling_rate[eye_channel[0]]

        eye_x = list()
        eye_y = list()
        eye_t = list()
        for idx in data.index:
            eye_x.append(data[eye_channel[0]].loc[idx])
            eye_y.append(data[eye_channel[1]].loc[idx])
            temp_num = data[eye_channel[0]].loc[idx].shape[0]
            temp_time = np.linspace(0, temp_num*samp_time, temp_num, endpoint=False)
            eye_t.append(temp_time)
        x_distance = [ite[eye_speed_win:]-ite[:-eye_speed_win] for ite in eye_x]
        y_distance = [ite[eye_speed_win:]-ite[:-eye_speed_win] for ite in eye_y]
        eye_speed = list()
        eye_freq = 1000.0/samp_time
        tim_dur = eye_speed_win*1.0/eye_freq
        for idx in self.data_df.index:
            speed = (x_distance[idx]**2 + y_distance[idx]**2)**0.5/tim_dur
            eye_speed.append(speed)
        eye_speed_t = [ite[eye_speed_win:] for ite in eye_t]
        eye_x_speed_pos = [ite[eye_speed_win:] for ite in eye_x]
        eye_y_speed_pos = [ite[eye_speed_win:] for ite in eye_y]
        # find saccade
        def saccade_find(trial_i):
            sac_half_speed_threshold = sac_speed_threshold/2.0
            tem_1 = np.where(eye_speed[trial_i] > sac_speed_threshold)[0]
            tem_2 = tem_1[1:] - tem_1[:-1]
            tem_3 = np.where(tem_2 > 1)[0]
            tem_3 = np.append(0, tem_3)
            tem_3 = np.append(tem_3, tem_2.shape[0])
            tem_4 = tem_3[1:] - tem_3[:-1]
            
            tem_5 = np.where(
                tem_4 > sac_duration_threshold * eye_freq / 1000.0)[0]
            tem_3_new = tem_3 + 1
            saccade_start = []
            saccade_end = []
            saccade_from = []
            saccade_to = []
            saccade_amp = []
            for tem_sac in tem_5:
                tem_6 = tem_3_new[tem_sac:tem_sac + 2]
                tem_7 = [tem_1[tem_6[0]], tem_1[tem_6[1] - 1]]
                tem_8 = [np.arange(tem_7[0]), np.arange(
                    tem_7[1], eye_speed[trial_i].shape[0])]

                if len(np.where(eye_speed[trial_i][tem_8[0]] < sac_half_speed_threshold)[0]) > 0:
                    sac_start = np.where(eye_speed[trial_i][
                                         tem_8[0]] < sac_half_speed_threshold)[0][-1]
                    
                else:
                    sac_start = 0

                if len(np.where(eye_speed[trial_i][tem_8[1]] < sac_half_speed_threshold)[0]) > 0:
                    sac_end = np.where(eye_speed[trial_i][tem_8[1]] < sac_half_speed_threshold)[
                        0][0] + tem_7[1]
                else:
                    sac_end = eye_speed[trial_i].shape[0] - 1
                if (sac_end - sac_start) * 1000.0 / eye_freq >= sac_duration_threshold:
                    sac_amp = ((eye_x_speed_pos[trial_i][sac_end] - eye_x_speed_pos[trial_i][sac_start])**2 + (
                        eye_y_speed_pos[trial_i][sac_end] - eye_y_speed_pos[trial_i][sac_start])**2)**0.5
                    if sac_amp > sac_displacement_threshold:
                        saccade_start.append(eye_speed_t[
                                             trial_i][sac_start])
                        saccade_end.append(eye_speed_t[
                                           trial_i][sac_end])
                        saccade_from.append(
                            [eye_x_speed_pos[trial_i][sac_start], eye_y_speed_pos[trial_i][sac_start]])
                        saccade_to.append(
                            [eye_x_speed_pos[trial_i][sac_end], eye_y_speed_pos[trial_i][sac_end]])
                        saccade_amp.append(sac_amp)
            return [saccade_start, saccade_end, saccade_from, saccade_to, saccade_amp]
            
        eye_saccade = [saccade_find(i) for i in range(len(eye_speed))]
        saccade_start = [np.array(ite[0]) for ite in eye_saccade]
        saccade_end = [np.array(ite[1]) for ite in eye_saccade]
        saccade_from = [np.array(ite[2]) for ite in eye_saccade]
        saccade_to = [np.array(ite[3]) for ite in eye_saccade]
        saccade_amp = [np.array(ite[4]) for ite in eye_saccade]
        self.data_df['saccade_start'] = saccade_start
        self.data_df['saccade_end'] = saccade_end
        self.data_df['saccade_from'] = saccade_from
        self.data_df['saccade_to'] = saccade_to
        self.data_df['saccade_amp'] = saccade_amp

    # choose saccades in each trial that happened within a certain period and of certain amplitude
    def choose_saccade(self, align_to, timebin, ampbin=False):
        '''
        Args
            align_to (string):
                event marker as zero point time
            timebin (list):
                time period relative to the zero point time
                Select saccades happened within the set period
            ampbin (list):
                amplitude range
                Selec saccades of set amplitude
                Default: False
        Return:
            -
        '''
        saccade_start = list()
        saccade_end = list()
        saccade_from = list()
        saccade_to = list()
        saccade_amp = list()
        for idx in self.data_df.index:
            sac_ids = range(len(self.data_df['saccade_start'].loc[idx]))
            temp_sac_start = list()
            temp_sac_end = list()
            temp_sac_from = list()
            temp_sac_to = list()
            temp_sac_amp = list()
            # time period relative to align_to marker
            timebin_0 = self.data_df[align_to].loc[idx] + timebin[0]
            timebin_1 = self.data_df[align_to].loc[idx] + timebin[1]
            for sac_id in sac_ids:
                if timebin_0 < self.data_df['saccade_start'].loc[idx][sac_id] < timebin_1:
                    if ampbin is not False:
                        if ampbin[0]<self.data_df['saccade_amp'].loc[idx][sac_id]<ampbin[1]:
                            temp_sac_start.append(self.data_df['saccade_start'].loc[idx][sac_id])
                            temp_sac_end.append(self.data_df['saccade_end'].loc[idx][sac_id])
                            temp_sac_from.append(self.data_df['saccade_from'].loc[idx][sac_id])
                            temp_sac_to.append(self.data_df['saccade_to'].loc[idx][sac_id])
                            temp_sac_amp.append(self.data_df['saccade_amp'].loc[idx][sac_id])
                    else:
                        temp_sac_start.append(self.data_df['saccade_start'].loc[idx][sac_id])
                        temp_sac_end.append(self.data_df['saccade_end'].loc[idx][sac_id])
                        temp_sac_from.append(self.data_df['saccade_from'].loc[idx][sac_id])
                        temp_sac_to.append(self.data_df['saccade_to'].loc[idx][sac_id])
                        temp_sac_amp.append(self.data_df['saccade_amp'].loc[idx][sac_id])
                            
            saccade_start.append(temp_sac_start)
            saccade_end.append(temp_sac_end)
            saccade_from.append(temp_sac_from)
            saccade_to.append(temp_sac_to)
            saccade_amp.append(temp_sac_amp)
        self.data_df['saccade_start_1'] = saccade_start
        self.data_df['saccade_end_1'] = saccade_end
        self.data_df['saccade_from_1'] = saccade_from
        self.data_df['saccade_to_1'] = saccade_to
        self.data_df['saccade_amp_1'] = saccade_amp

    # Reallocate the storage space that the occupied by the file, then release extra storage space.
    def reclaim_space(self,file_name):
        '''
        Args
            file_name (string):
                the name of the work space 
        Return
            -
        '''
        f = hp.File(file_name,'r')
        f2 = hp.File(file_name.split('.h5')[0]+'_reclaim.h5','w')
        used_keys = list()
        def valid_key(name):
            if isinstance(f[name],hp.Group):
                pass
            else:
                used_keys.append(name)
        f.visit(valid_key)
        for key in used_keys:
            f2[key] = f[key].value
        f.flush()
        f2.flush()
        f.close()
        f2.close()
        os.remove(file_name)
        os.rename(file_name.split('.h5')[0]+'_reclaim.h5',file_name)
        print('Space is reclaimed now')
