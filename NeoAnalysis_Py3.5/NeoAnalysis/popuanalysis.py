# -*- coding: utf-8 -*-
'''
The Module for analyzing data at population level.
This module uses the results stored in the workspace obtained from analyzing single session data.
This work is based on:
    * Bo Zhang, Ji Dai - first version
'''
import h5py as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from . import func_tools as ftl
import scipy
import seaborn
seaborn.set_style(style='white')

class PopuAnalysis():
    '''
    class for analyzing data at population level.
    
    Args
        filename (string):
            file name of the workspace (with extension)
    Results
        -
    Examples
        >>> PopuAnalysis('test_workspace.h5')
        initiate PopuAnalysis class
    '''
    def __init__(self,filename):
        """
        Initialize the PopuAnalysis class.
        """
        store = hp.File(filename)
        self.store = store
    
    # normalize firing rate using min-max normalization
    def __spike_normalize(self,data):
        return_data = list()
        for ite in data:
            max_data = max([value.max() for _,value in ite.items()])
            min_data = min([value.min() for _,value in ite.items()])
            norm_data = {key:(value-min_data)/(max_data-min_data) for key,value in ite.items()}
            return_data.append(norm_data)
        return return_data
    
    # plot average PSTH among neuronal population
    def plot_spike(self,store_key,conditions,normalize=False,fig_mark=[0],line_style=False,x_lim=False,y_lim=False,err_style='ci_band',ci=68):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            conditions (list):
                define spikes of which experimental conditions will be plotted
            normalize (Boolean, list):
                if True, min-max normalization will be used among all conditions
                if False, no normalization
                if list, min-max normalization will be used among conditions in the list
                Default: False
            fig_mark (list):
                Draw vertical lines at the time points in the list.
                Default: [0]
            line_style (list):
                line style used for plotting. 
                the length of the line_style list must equal the length of the conditions
                Default: False, automatically use line styles for different lines
            x_lim (list):
                set limits of x-axis
                Default: False
            y_lim (list):
                set limits of y-axis
                Default: False
            err_style (string):
                Set how to plot the uncertainty across units, select from {ci_band, ci_bars, boot_traces,
                boot_kde, unit_traces, unit_points}
                Default: ci_band
            ci (int):
                confidence interval.
                Default: 68
        Returns
            {'data': {condition_1:PSTH,
                      condition_2:PSTH,
                      .
                      .
                      .},
             'time':firing rate time}
        '''
        fr_time = self.store[list(self.store.keys())[0]][store_key]['time'].value
        if isinstance(normalize,list):
            # check whether conditions are subset of normalize
            for ite in conditions:
                if ite not in normalize:
                    raise ValueError("conditions should be subset of normalize parameter")
            datas_all = list()
            for ite_file in list(self.store.keys()):
                data_temp = dict()
                for ite_cond in normalize:
                    data_temp[ite_cond] = self.store[ite_file][store_key]['data'][str(ite_cond)].value
                datas_all.append(data_temp)
            datas_normalize = self.__spike_normalize(datas_all)
        elif normalize is True:
            datas_all = list()
            for ite_file in list(self.store.keys()):
                data_temp = dict()
                for ite_cond in list(self.store[ite_file][store_key]['data'].keys()):
                    if (")" in ite_cond) & ("(" in ite_cond):
                        ite_cond = eval(ite_cond, dict(__builtins__=None))
                    data_temp[ite_cond] = self.store[ite_file][store_key]['data'][str(ite_cond)].value
                datas_all.append(data_temp)
            datas_normalize = self.__spike_normalize(datas_all)
        elif normalize is False:
            datas_all = list()
            for ite_file in list(self.store.keys()):
                data_temp = dict()
                for ite_cond in conditions:
                    data_temp[ite_cond] = self.store[ite_file][store_key]['data'][str(ite_cond)].value
                datas_all.append(data_temp)
            datas_normalize = datas_all
        else:
            raise ValueError("normalize parameter type should be list of condition, True or False")

        target_mean = dict()
        targets = dict()
        for key in conditions:
            temp_data = [value[key] for value in datas_normalize]
            temp_data = np.array(temp_data)
            targets[key] = temp_data
            target_mean[key] = temp_data.mean(axis=0)

        fr_max = max([v.max() for _,v in target_mean.items()])

        plt.figure(figsize=[12,8])
        legends = []
        for i, key in enumerate(conditions):
            seaborn.tsplot(data=targets[key],time=fr_time,color=seaborn.color_palette()[i],linewidth=2,err_style=err_style,ci=ci)
            legends.append(mlines.Line2D([],[],color=seaborn.color_palette()[i],label=key,linewidth=2))
        plt.legend(handles=legends,fontsize=12,framealpha=0,labelspacing=0)

        if x_lim is False:
            x_lim = [fr_time[0],fr_time[-1]]
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xlabel('Time (ms)',fontsize=16)
        if normalize is False:
            plt.ylabel('Mean firing rate (spikes/s)',fontsize=16)
        else:
            plt.ylabel('Normalized mean firing rate',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.vlines(fig_mark,0,fr_max)
        return {'data':target_mean, 'time':fr_time}

    # plot spectrogram of analog signals (e.g. LFP) at population level
    def plot_spectrogram(self,store_key,condition,fig_mark=[0],y_lim=[0,100]):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            condition (string):
                define which conditions will be plotted
            fig_mark (list):
                Draw vertical lines at the time points in the list.
                Default: [0]
            y_lim (list):
                set limits of y-axis
                Default: [0,100]
        Returns
            {'frequency':frequency,
             'time':analog signal time,
             'data':spectrogram value}     
        '''
        time = self.store[list(self.store.keys())[0]][store_key]['time'].value
        freq = self.store[list(self.store.keys())[0]][store_key]['frequency'].value
        datas_all = list()
        for ite_file in list(self.store.keys()):
            datas_all.append(self.store[ite_file][store_key]['data'][str(condition)].value)
        datas_all = np.array(datas_all)
        specgram = datas_all.mean(axis=0)
        vmin = specgram[(y_lim[0]<freq) & (freq<y_lim[1])].min()
        vmax = specgram[(y_lim[0]<freq) & (freq<y_lim[1])].max()
        halfbin_time = (time[1] - time[0]) / 2.0
        halfbin_freq = (freq[1] - freq[0]) / 2.0
        specgram_1 = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        plt.figure()
        plt.imshow(specgram_1, interpolation="nearest", extent=extent,cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        plt.axis('tight')
        fig_mark = [ite/1000.0 for ite in fig_mark]
        if y_lim is not False:
            plt.ylim(y_lim)
            plt.vlines(fig_mark,y_lim[0],y_lim[1],color='r')
        plt.colorbar()
        plt.title(str(condition))
        return {'frequency':freq,'time':time,'data':specgram}

    # plot scalar data (e.g. reaction time) in population level
    def plot_line(self,store_key,conditions):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            conditions (list):
                define which conditions will be plotted
        Returns
            {'mean':{'condition_1':value,
                     'condition_2':value,
                     .
                     .
                     .},
            'sem':{'condition_1':value,
                   'condition_2':value,
                   .
                   .
                   .}
            }
        '''
        datas_all = list()
        for ite_file in list(self.store.keys()):
            temp_data = {cond:self.store[ite_file][store_key][cond]['mean'].value for cond in list(self.store[ite_file][store_key].keys())}
            datas_all.append(temp_data)
        target_mean = dict()
        target_sem = dict()
        
        for cond in conditions:
            cond = str(cond)
            temp_data = [ite[cond] for ite in datas_all]
            temp_data = pd.Series(temp_data)
            target_sem[cond] = temp_data.sem()
            target_mean[cond] = temp_data.mean()
        
        if isinstance(conditions[0],tuple) and len(conditions[0]) == 2:
            conds_1 = sorted(list(set(ite[0] for ite in conditions)))
            conds_2 = sorted(list(set(ite[1] for ite in conditions)))
            for cond_2 in conds_2:
                xs = list()
                xticks = list()
                ys = list()
                err = list()
                for i,cond_1 in enumerate(conds_1):
                    if (cond_1,cond_2) in conditions:
                        ys.append(target_mean[str(tuple([cond_1,cond_2]))])
                        err.append(target_sem[str(tuple([cond_1,cond_2]))])
                        xs.append(i)
                        xticks.append(cond_1)
                plt.errorbar(xs,ys,yerr=err,label=str(cond_2))
                plt.xlim(xs[0]-1,xs[-1]+1)
                plt.xticks(xs,xticks)
                plt.legend(framealpha=0,labelspacing=0.01)
        elif isinstance(conditions[0],str):
            columns = sorted(conditions)
            conds_1 = columns
            xs = list()
            xticks = list()
            ys = list()
            err = list()
            for i,cond_1 in enumerate(conds_1):
                if cond_1 in columns:
                    ys.append(target_mean[cond_i])
                    err.append(target_sem[cond_i])
                    xs.append(i)
                    xticks.append(cond_1)
            plt.errorbar(xs,ys,yerr=err)
            plt.xlim(xs[0]-1,xs[-1]+1)
            plt.xticks(xs,xticks)
        return {'mean':target_mean, 'sem':target_sem}
    
    # plot scalar data (e.g. reaction time) in population level
    def plot_bar(self,store_key,conditions,ci=95,kind='bar'):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            conditions (list):
                define which conditions will be plotted
            ci (float):
                confidence interval
                defaule: 95
            kind (str):
                The kind of plot to draw., link 'bar', 'point'
                Default: 'bar'
        Returns
            {'mean':{'condition_1':value,
                     'condition_2':value,
                     .
                     .
                     .},
            'sem':{'condition_1':value,
                   'condition_2':value,
                   .
                   .
                   .}
            }
        '''
        datas_all = list()
        for ite_file in self.store.keys():
            temp_data = {cond:self.store[ite_file][store_key][cond]['mean'].value for cond in self.store[ite_file][store_key].keys()}
            datas_all.append(temp_data)

        target_mean = dict()
        target_sem = dict()        
        for cond in conditions:
            cond = str(cond)
            temp_data = [ite[cond] for ite in datas_all]
            temp_data = pd.Series(temp_data)
            target_sem[cond] = temp_data.sem()
            target_mean[cond] = temp_data.mean()

        if isinstance(conditions[0],tuple) and len(conditions[0]) == 2:
            cond_1 = []
            cond_2 = []
            values = []
            for ite in datas_all:
                for k,v in ite.items():
                    k = eval(k)
                    cond_1.append(k[0])
                    cond_2.append(k[1])
                    values.append(v)
            datas_all = pd.DataFrame({'condition_1':cond_1,'condition_2':cond_2,'values':values})
            g = seaborn.factorplot(x='condition_1',y='values',hue='condition_2',data=datas_all,size=6,kind=kind,palette='muted',ci=ci,
                                    order=sorted(list(set(datas_all['condition_1'].values))),hue_order=sorted(list(set(datas_all['condition_2'].values))))
            g.set_ylabels('')
        elif isinstance(conditions[0],str):
            cond_1 = []
            values = []
            for ite in datas_all:
                for k,v in ite.items():
                    cond_1.append(k)
                    values.append(v)
            datas_all = pd.DataFrame({'condition_1':cond_1,'values':values})
            g = seaborn.factorplot(x='condition_1',y='values',data=datas_all,size=6,kind=kind,palette='muted',ci=ci,
                                    order=sorted(list(set(datas_all['condition_1'].values))))
            g.set_ylabels('')
            
            # print datas_all
        '''
        target_mean = dict()
        target_sem = dict()        
        for cond in conditions:
            cond = str(cond)
            temp_data = [ite[cond] for ite in datas_all]
            temp_data = pd.Series(temp_data)
            target_sem[cond] = temp_data.sem()
            target_mean[cond] = temp_data.mean()

        if isinstance(conditions[0],tuple) and len(conditions[0]) == 2:
            conds_1 = sorted(list(set(ite[0] for ite in conditions)))
            conds_2 = sorted(list(set(ite[1] for ite in conditions)))
            for cond_2 in conds_2:
                xs = list()
                xticks = list()
                ys = list()
                err = list()
                for i,cond_1 in enumerate(conds_1):
                    if (cond_1,cond_2) in conditions:
                        ys.append(target_mean[str(tuple([cond_1,cond_2]))])
                        err.append(target_sem[str(tuple([cond_1,cond_2]))])
                        xs.append(i)
                        xticks.append(cond_1)
                plt.errorbar(xs,ys,yerr=err,label=str(cond_2))
                plt.xlim(xs[0]-1,xs[-1]+1)
                plt.xticks(xs,xticks)
                plt.legend(framealpha=0,labelspacing=0.01)
        elif isinstance(conditions[0],str):
            columns = sorted(conditions)
            conds_1 = columns
            xs = list()
            xticks = list()
            ys = list()
            err = list()
            for i,cond_1 in enumerate(conds_1):
                if cond_1 in columns:
                    ys.append(target_mean[cond_i])
                    err.append(target_sem[cond_i])
                    xs.append(i)
                    xticks.append(cond_1)
            plt.errorbar(xs,ys,yerr=err)
            plt.xlim(xs[0]-1,xs[-1]+1)
            plt.xticks(xs,xticks)
        '''
        return {'mean':target_mean, 'sem':target_sem}

    # plot analog signals (e.g. LFP) at population level
    def plot_analog(self,store_key,conditions,line_style=False,fig_mark=[0],x_lim=False,y_lim=False):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            conditions (list):
                define which conditions will be plotted
            line_style (list):
                line style used for plotting. 
                the length of the line_style list must equal the length of conditions
                Default: False, automatically use line styles for different lines
            fig_mark (list):
                Draw vertical lines at the time points in the list.
                Default: [0]
            x_lim (list):
                set limits of y-axis
                Default: False
            y_lim (list):
                set limits of y-axis
                Default: False
        Returns
            {'time':time,'data':{'condition_1':mean signal data,
                                 'condition_2':mean signal data,
                                  .
                                  .
                                  .}
            }
        '''
        times = self.store[list(self.store.keys())[0]][store_key]['time'].value
        datas_all = list()
        for ite_file in list(self.store.keys()):
            data_temp = dict()
            for ite_cond in conditions:
                data_temp[ite_cond] = self.store[ite_file][store_key]['data'][str(ite_cond)].value
            datas_all.append(data_temp)
        target_mean = dict()
        for key in conditions:
            temp_data = [value[key] for value in datas_all]
            temp_data = np.array(temp_data)
            target_mean[key] = temp_data.mean(axis=0)

        target_max = max([v.max() for _,v in target_mean.items()])
        target_min = min([v.min() for _,v in target_mean.items()])
        plt.figure(figsize=[12,8])
        for i,key in enumerate(conditions):
            if line_style is not False:
                plt.plot(times, target_mean[key],line_style[i],linewidth=2,label=key)
            else:
                plt.plot(times, target_mean[key],linewidth=2,label=key)
        if x_lim is False:
            x_lim = [times[0],times[-1]]
        if y_lim is False:
            y_lim = [target_min-10, target_max+10]
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xlabel('Time (ms)',fontsize=16)
        plt.ylabel('LFP (mV)',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.vlines(fig_mark,target_min,target_max)
        plt.legend(framealpha=0,labelspacing=0)
        return {'time':times,'data':target_mean}

    # plot spectrum of analog signals (e.g. LFP) in population level
    def plot_spectral(self,store_key,conditions,line_style=False,x_lim=[0,100],log=False):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            conditions (list):
                define which conditions will be plotted
            line_style (list):
                line style used for plotting. 
                the length of the line_style list must equal the length of conditions
                Default: False, automatically use line styles for different lines
            x_lim (list):
                set limits of y-axis
                Default: [0,100]
            log (Boolean):
                if True, y-axis will use logarithmic axis
                Default: False
        Returns
            {'frequency':frequency, 
             'data':{'condition_1':mean signal data,
                     'condition_2':mean signal data,
                     .
                     .
                     .}}
        '''
        freq = self.store[list(self.store.keys())[0]][store_key]['frequency'].value
        datas_all = list()
        for ite_file in list(self.store.keys()):
            data_temp = dict()
            for ite_cond in conditions:
                data_temp[ite_cond] = self.store[ite_file][store_key]['data'][str(ite_cond)].value
            datas_all.append(data_temp)

        target_mean = dict()
        for key in conditions:
            temp_data = [value[key] for value in datas_all]
            temp_data = np.array(temp_data)
            target_mean[key] = temp_data.mean(axis=0)

        target_max = max([v.max() for _,v in target_mean.items()])
        target_min = min([v.min() for _,v in target_mean.items()])
        plt.figure(figsize=[12,8])
        for i,key in enumerate(conditions):
            if log is False:
                if line_style is not False:
                    plt.plot(freq, target_mean[key],line_style[i],linewidth=2,label=key)
                else:
                    plt.plot(freq, target_mean[key],linewidth=2,label=key)
            elif log is True:
                if line_style is not False:
                    plt.semilogy(freq, target_mean[key],line_style[i],linewidth=2,label=key)
                else:
                    plt.semilogy(freq, target_mean[key],linewidth=2,label=key)
                
        if x_lim is not False:
            plt.xlim(x_lim)
        plt.xlabel('Frequency (Hz)',fontsize=16)
        plt.ylabel(' ',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(framealpha=0,labelspacing=0)
        return {'frequency':freq,'data':target_mean}

    # paired t-test 
    # for scalar value usage only
    def stats_ttest_rel(self,store_key,cond_1,cond_2):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            cond_1 (string), cond_2 (string):
                compare these two conditions using paired t-test 
        Return
            pvalue
        '''
        datas_1 = list()
        datas_2 = list()
        for ite_file in list(self.store.keys()):
            datas_1.append(self.store[ite_file][store_key][str(cond_1)]['mean'].value)
            datas_2.append(self.store[ite_file][store_key][str(cond_2)]['mean'].value)
        datas_1 = np.array(datas_1)
        datas_2 = np.array(datas_2)
        return scipy.stats.ttest_rel(datas_1,datas_2).pvalue

    # t-test 
    # for scalar value usage only
    def stats_ttest_ind(self,store_key,cond_1,cond_2):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            cond_1 (string), cond_2 (string):
                compare these two conditions using t-test 
        Returns
            pvalue
        '''
        datas_1 = list()
        datas_2 = list()
        for ite_file in list(self.store.keys()):
            datas_1.append(self.store[ite_file][store_key][str(cond_1)]['mean'].value)
            datas_2.append(self.store[ite_file][store_key][str(cond_2)]['mean'].value)
        datas_1 = np.array(datas_1)
        datas_2 = np.array(datas_2)
        return scipy.stats.ttest_ind(datas_1,datas_2).pvalue

    # calculate t-test for the mean of one group of scores
    # for scalar value usage only
    def stats_ttest_1samp(self,store_key,cond,compare_value):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            cond (string):
                sample observation
            compare_value (float):
                expected value in null hypothesis
        Returns
            pvalue
        '''
        datas = list()
        for ite_file in list(self.store.keys()):
            datas.append(self.store[ite_file][store_key][str(cond)]['mean'].value)
        datas = np.array(datas)
        return scipy.stats.ttest_1samp(datas,compare_value).pvalue

    # descriptive statistics
    # for scalar value usage only
    def stats_desc(self,store_key,cond):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            cond (string):
                sample observation
        Returns
            descriptive statistics
        '''
        datas = list()
        for ite_file in list(self.store.keys()):
            datas.append(self.store[ite_file][store_key][str(cond)]['mean'].value)
        datas = pd.Series(datas)
        return datas.describe()

    # one way ANOVA
    # for scalar value usage only
    def stats_anova_oneway(self,store_key,conditions):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            conditions (list):
                list of experimental condition names
        Returns
            pvalue
        '''
        datas = list()
        for cond in conditions:
            temp_data = list()
            for ite_file in list(self.store.keys()):
                temp_data.append(self.store[ite_file][store_key][cond]['mean'].value)
            datas.append(np.array(temp_data))
        return scipy.stats.f_oneway(*datas).pvalue

    # two way ANOVA
    # for scalar value usage only
    def stats_anova_twoway(self,store_key,conditions):
        '''
        Args
            store_key (string):
                define which data to be analyzed in the workspace
            conditions (list):
                list of experimental condition names
        Returns
            pvalue
        '''
        datas = list()
        keys = list()
        for ite_file in list(self.store.keys()):
            for cond in conditions:
                datas.append(self.store[ite_file][store_key][str(cond)]['mean'].value)
                keys.append(cond)
        key_1 = [ite[0] for ite in keys]
        key_2 = [ite[1] for ite in keys]
        target_data = pd.DataFrame({'data':datas,'factor_1':key_1,'factor_2':key_2})
        formula = 'data~C(factor_1)+C(factor_2)+C(factor_1):C(factor_2)'
        lm = ols(formula,target_data).fit()
        anovaResults = anova_lm(lm)
        return anovaResults

    # close the work space
    def close(self):
        self.store.close()