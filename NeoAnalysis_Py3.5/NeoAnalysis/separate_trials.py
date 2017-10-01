import numpy as np
import pandas as pd
from . import func_tools as ftl
def separate_trials(events,comments,spikes,analogs,trial_start_mark,comment_expr):
    trial_separate_time = events['times'][events['labels']==trial_start_mark]
    trial_index = np.digitize(trial_separate_time,trial_separate_time)
    rawdata = pd.DataFrame()
    # separate events
    events_unique = np.unique(events['labels'])
    for key in events_unique:
        value_mask = events['labels'] == key
        temp_index = np.digitize(events['times'][value_mask],trial_separate_time)
        temp_times = [events['times'][value_mask][temp_index==i_trial] for i_trial in trial_index]
        temp_times = list(map(ftl.one_elem_list ,temp_times))

        rawdata['event_'+key] = temp_times
    # align events (not contains trial_start_mark now) 
    for key in rawdata.keys():
        if key != 'event_'+str(trial_start_mark):
            rawdata[key] = rawdata[key] - rawdata['event_'+str(trial_start_mark)]
    # expr comments
    comment_separate = comment_expr.replace("key",'').replace("value",'')
    comment_key_id = comment_expr.split(comment_separate).index("key")
    comment_value_id = comment_expr.split(comment_separate).index("value")
    temp_func = lambda ite:[ite.split(comment_separate)[comment_key_id],ite.split(comment_separate)[comment_value_id]]
    labels_key_value = list(map(temp_func,comments['labels']))
    labels_key_value = np.array(labels_key_value)
    comments_unique = np.unique(labels_key_value[:,0])
    for key in comments_unique:
        value_mask = labels_key_value[:,0]==key
        temp_index = np.digitize(comments['times'][value_mask],trial_separate_time)
        temp_value = [labels_key_value[:,1][value_mask][temp_index==i_trial] for i_trial in trial_index]
        temp_value = list(map(ftl.one_elem_list,temp_value))
        rawdata[str(key)] = temp_value
    # separate spikes
    for chn in spikes.keys():
        temp_index = np.digitize(spikes[chn],trial_separate_time)
        temp_value = [spikes[chn][temp_index==i_trial] for i_trial in trial_index]
        rawdata[chn] = temp_value
        rawdata[chn] = rawdata[chn] - rawdata['event_'+str(trial_start_mark)]
    # separate analogs
    for chn in analogs.keys():
        datashape = analogs[chn]['data'].shape[0]
        start_time = analogs[chn]['start_time']
        sampling_rate = analogs[chn]['sampling_rate']
        times = np.linspace(start_time,start_time+1000.0*datashape/sampling_rate,datashape,endpoint=False)
        temp_index = np.digitize(times,trial_separate_time)
        temp_datas = [analogs[chn]['data'][temp_index==i_trial] for i_trial in trial_index]
        rawdata[chn] = temp_datas
    rawdata['event_'+str(trial_start_mark)] = 0.0
    return rawdata
    # for chn in spikes.keys():
    #     unit_unique = np.unique(spikes[chn]['units'])
    #     for unit in unit_unique:
    #         value_mask = spikes[chn]['units'] == unit
    #         temp_index = np.digitize(spikes[chn]['times'][value_mask],trial_separate_time)
    #         temp_value = [spikes[chn]['times'][value_mask][temp_index==i_trial] for i_trial in trial_index]
    #         rawdata[chn+'_'+str(unit)] = temp_value
    #         rawdata[chn+'_'+str(unit)] = rawdata[chn+'_'+str(unit)] - rawdata['dig_'+str(trial_start_mark)]
    # # separate analogs
    # analog_samp_period = dict()
    # for chn in analogs.keys():
    #     samp_period = np.diff(analogs[chn]['times']).mean()
    #     analog_samp_period[chn] = round(samp_period,3)
    #     temp_index = np.digitize(analogs[chn]['times'],trial_separate_time)
    #     temp_datas = [analogs[chn]['datas'][temp_index==i_trial] for i_trial in trial_index]
    #     temp_times = [analogs[chn]['times'][temp_index==i_trial][0] for i_trial in trial_index]
    #     rawdata[chn] = temp_datas
    #     rawdata._metadata = ['samp_period']
    # if len(analog_samp_period)>0:
    #     rawdata.samp_period = analog_samp_period
    # else:
    #     rawdata.samp_period = dict()
    # rawdata['dig_'+str(trial_start_mark)] = 0.0
    # return rawdata