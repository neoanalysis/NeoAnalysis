# -*- coding: utf-8 -*-
'''
This is a module for filtering analog signal. The filter methods include band-pass and band-stop filtering.
This module can be used with or without a GUI window.
This work is based on:
    * Bo Zhang, Ji Dai - first version
'''
import copy
from . import pyqtgraph as pg
from .pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .pyqtgraph import opengl as gl
import numpy as np
import os
import h5py as hp
import quantities as pq
import sys
from . import func_tools as ftl

class AnalogFilter():
    '''
    This is a Class for filtering analog signal. 
    Two filtering methods are provided: band pass filtering and band stop filtering.
    This class provides an optional GUI window to show the filtering results instantaneously.

    Args
        gui (Boolean):
            If true, a GUI window will be displayed, in which users can selecte the analog channels 
            and filtering methods to process the data.
            Default: True
        reclaim_space (Boolean):
            If true, the storage space that the file occupied will be reallocated, which will realsease
            extra storage space.
            Default: False
        filename (string):
            The filename (including the file path) for analysis.
            When gui = False, users need to set the filename (with extension).
        channels (str or list):
            The channel for filtering.
            When gui = False, users need to set the channel.            
        bandpass (list):
            When gui = False, users need to set the bandpass.
            Set band pass value for signal filtering.
        bandstop (list):
            When gui = False, users need to set the bandstop.
            Set band stop value for signal filtering.
    Returns
        -
    Examples
        >>> AnalogFilter(False,False,'myfile.h5',['analog_23','analog_26'],[4,100],[59,61])
            #Use band-pass (4-100 Hz) and band-stop (59-61 Hz) to filter signal in analog_23 and analog_26 channels in the file 'myfile.h5'
    '''
    def __init__(self,gui=True,reclaim_space=False,filename=None,channels=None,bandpass=None,bandstop=None):
        """
        Initialize the AnalogFilter class.
        """
        # A gui window will be show if gui = True
        if gui is True:
            self.reclaim_space = reclaim_space
            # create the main GUI window
            app = QtGui.QApplication([])
            win = pg.GraphicsWindow()
            win.resize=(1200,800)
            self.colors=[(0,0,1*255,255),(1*255,1*255,1*255,255),(0,1*255,0,255),(1*255,0,0,255),(1*255,1*255,0,255),
                        (1*255,0,1*255,255),(0,1*255,1*255,255),(0,0.5*255,0.5*255,255),(0.5*255,0,0.5*255,255),(0.5*255,0.5*255,0,255)]
            # Three sub-windows: p1, p2, p3
            # p1: draw selected analog signal 
            # p2: draw various control buttons
            # p3: draw all analog signal in a certain channel
            self.p1 = win.addPlot(row=0,col=0)
            self.p1.setMaximumHeight(600)
            self.p1.setMaximumWidth(1000)
            self.p2 = win.addPlot(row=1,col=0)
            self.p2.setMaximumHeight(200)
            self.p2.setMaximumWidth(1000)
            self.p2.setMouseEnabled(x=True,y=False)

            self.p3 = win.addPlot(row=0,col=1,rowspan=2)
            self.p3.setMaximumHeight(800)
            self.p3.setMaximumWidth(150)
            self.p3.hideAxis('bottom')
            self.p3.hideAxis('left')
            self.p3.setXRange(-1.5,1.5)
            self.p3.setYRange(0,30)
            
            self.is_linked = False
            # set load button in p3
            p3_load_button = pg.ScatterPlotItem(symbol='s')
            p3_load_button.setData([-0.7],[3])
            p3_load_button.setSize(25)
            p3_load_button.setPen('g')
            self.p3.addItem(p3_load_button)
            p3_load_text = pg.TextItem('load',color='w')
            p3_load_text.setPos(-1.05,2.5)
            self.p3.addItem(p3_load_text)

            # set save button in p3
            self.p3_save_button = pg.ScatterPlotItem(symbol='s')
            self.p3_save_button.setData([0.7],[3])
            self.p3_save_button.setSize(25)
            self.p3_save_button.setPen('g')
            self.p3.addItem(self.p3_save_button)
            p3_save_text = pg.TextItem('save',color='w')
            p3_save_text.setPos(0.33,2.5)
            self.p3.addItem(p3_save_text)

            # set band_pass button in p3
            p3_bandpass_button = pg.ScatterPlotItem(symbol='s')
            p3_bandpass_button.setData([-0.7],[8])
            p3_bandpass_button.setSize(25)
            p3_bandpass_button.setPen('b')
            self.p3.addItem(p3_bandpass_button)
            p3_bandpass_text = pg.TextItem('band pass',color='w')
            p3_bandpass_text.setPos(-0.4,8.7)
            self.p3.addItem(p3_bandpass_text)
            self.p3_bandpass_button = p3_bandpass_button

            # set band_stop button in p3
            p3_bandstop_button = pg.ScatterPlotItem(symbol='s')
            p3_bandstop_button.setData([-0.7],[10])
            p3_bandstop_button.setSize(25)
            p3_bandstop_button.setPen('b')
            self.p3.addItem(p3_bandstop_button)
            p3_bandstop_text = pg.TextItem('band stop',color='w')
            p3_bandstop_text.setPos(-0.4,10.7)
            self.p3.addItem(p3_bandstop_text)
            self.p3_bandstop_button = p3_bandstop_button

            # set reset button in p3
            p3_reset_button = pg.ScatterPlotItem(symbol='s')
            p3_reset_button.setData([-0.7],[6])
            p3_reset_button.setSize(25)
            p3_reset_button.setPen('b')
            self.p3.addItem(p3_reset_button)
            p3_reset_text = pg.TextItem('reset',color='w')
            p3_reset_text.setPos(-0.4,6.7)
            self.p3.addItem(p3_reset_text)
            self.p3_reset_button = p3_reset_button

            self.p3_item_raw_num = len(self.p3.items)
            p3_load_button.sigClicked.connect(self.__load_data)
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtGui.QApplication.instance().exec_()

        # Do not load a GUI window
        elif gui is False:
            with hp.File(filename,'a') as f:
                if isinstance(channels,str) and (channels in f['analogs'].keys()):
                    if bandpass is not None:
                        data = f['analogs'][channels]['data'].value
                        sampling_rate = f['analogs'][channels]['sampling_rate'].value
                        result_data = self.__filter_bandpass_nogui(data,bandpass,sampling_rate)
                        del f['analogs'][channels]['data']
                        f['analogs'][channels]['data'] = result_data
                    if bandstop is not None:
                        data = f['analogs'][channels]['data'].value
                        sampling_rate = f['analogs'][channels]['sampling_rate'].value
                        result_data = self.__filter_bandstop_nogui(data,bandpass,sampling_rate) 
                        del f['analogs'][channels]['data']
                        f['analogs'][channels]['data'] = result_data

                elif isinstance(channels,list):
                    for channel in channels:
                        if channel not in f['analogs'].keys():
                            raise ValueError('%s is not a valid channel'%channel)
                    for channel in channels:
                        if bandpass is not None:
                            data = f['analogs'][channel]['data'].value
                            sampling_rate = f['analogs'][channel]['sampling_rate'].value
                            result_data = self.__filter_bandpass_nogui(data,bandpass,sampling_rate)
                            del f['analogs'][channel]['data']
                            f['analogs'][channel]['data'] = result_data
                        if bandstop is not None:
                            data = f['analogs'][channel]['data'].value
                            sampling_rate = f['analogs'][channel]['sampling_rate'].value
                            result_data = self.__filter_bandstop_nogui(data,bandpass,sampling_rate) 
                            del f['analogs'][channel]['data']
                            f['analogs'][channel]['data'] = result_data
                else:
                    raise ValueError('parameter channels should be list or str type, or it is not a valid channel')

    def __load_data(self):
        # Load a file-selection dialog
        name = QtGui.QFileDialog.getOpenFileName(filter="hdf5 (*.h5)")
        self.file_name = name[0]
        if len(self.file_name) >0:
            self.p1.clear()
            self.p2.clear()
            self.p2_roi_pos1 = 0
            self.p2_roi_pos2 = 10000

            # if len(self.p3.items) == self.p3_item_raw_num:
            if self.is_linked is False:
                self.p3_bandpass_button.sigClicked.connect(self.__band_pass)
                self.p3_bandstop_button.sigClicked.connect(self.__band_stop)
                self.p3_reset_button.sigClicked.connect(self.__reset_popup)
                self.p3_save_button.sigClicked.connect(self.__save_popup)
                self.is_linked = True

            for item in self.p3.items[self.p3_item_raw_num:]:
                self.p3.removeItem(item)
            self.unique_chan = list()
            with hp.File(self.file_name,'r') as f:
                for key in f.keys():
                    if key == 'analogs':
                        for tem_key in f['analogs'].keys():
                            self.unique_chan.append(tem_key)
            # add channel button in p3
            self.p3_chns_button = [0]*len(self.unique_chan)
            self.p3_chns_text = [0]*len(self.unique_chan)
            for i,ch in enumerate(self.unique_chan):
                self.p3_chns_button[i] = pg.ScatterPlotItem()
                self.p3_chns_button[i].setPen('k')
                self.p3_chns_button[i].setData([-1.0],[30-1.3*i],size=16)
                self.p3_chns_text[i] = pg.TextItem(ch,color='w')
                self.p3_chns_text[i].setPos(-0.8,30-1.3*i+0.7)
                self.p3.addItem(self.p3_chns_button[i])
                self.p3.addItem(self.p3_chns_text[i])
            # set roi in p2, but not add to p2 now
            self.p2_roi = pg.LinearRegionItem()
            self.p2_roi.setRegion([self.p2_roi_pos1,self.p2_roi_pos2])
            self.p2_roi.setZValue(10)
            # add update func to p3_chns_button
            for i,ch in enumerate(self.unique_chan):
                self.p3_chns_button[i].sigClicked.connect(self.__update_channel(ch))
            # initiate a variable to storage the filtered signals
            self.analogs_pool = dict()

            self.select_chn = None
        elif len(self.file_name) == 0:
            print('No file is selected')
            return None

    def __update_channel(self,select_chn):
        # recording the analog signal channel used currently
        def click():
            self.select_chn = select_chn
            self.__add_to_memory(self.select_chn)
            start_time = self.analogs_pool[self.select_chn]['start_time']
            sampling_rate = self.analogs_pool[self.select_chn]['sampling_rate']
            datashape = self.analogs_pool[self.select_chn]['data'].shape[0]
            self.times = np.linspace(start_time,start_time+1000.0*datashape/sampling_rate,datashape,endpoint=False)
            print("%s is selected"%self.select_chn)
            for i,chn in enumerate(self.unique_chan):
                if chn == select_chn:
                    self.p3_chns_button[i].setPen('r')
                else:
                    self.p3_chns_button[i].setPen('k')
            # clean items in p2
            self.p2.clear()
            self.p2.addItem(self.p2_roi,ignoreBounds=True)
            self.p2_roi.setRegion([self.p2_roi_pos1, self.p2_roi_pos2])
            self.p2_roi.sigRegionChangeFinished.connect(self.__update_roi_p2)
            self.selected_indexs_p2 = np.where((self.times>self.p2_roi_pos1) & (self.times<self.p2_roi_pos2))[0]
            self.selected_indexs_p2 = np.array(self.selected_indexs_p2,dtype=np.int32)
            # draw signal
            self.__draw_p2()
            self.__draw_p1()
        return click
        
    def __add_to_memory(self,select_chn):
        # add data in new selected signal channel to variable self.analogs_pool
        if select_chn not in self.analogs_pool.keys():
            with hp.File(self.file_name,'r') as f:
                self.analogs_pool[select_chn] = dict()
                self.analogs_pool[select_chn]['data'] = f['analogs'][select_chn]['data'].value
                self.analogs_pool[select_chn]['sampling_rate'] = f['analogs'][select_chn]['sampling_rate'].value
                self.analogs_pool[select_chn]['start_time'] = f['analogs'][select_chn]['start_time'].value

    def __update_roi_p2(self):
        # update time bin selected widget in p2
        minX,maxX = self.p2_roi.getRegion()
        self.selected_indexs_p2 = np.where((self.times>minX) & (self.times<maxX))[0]
        self.selected_indexs_p2 = np.array(self.selected_indexs_p2,dtype=np.int32)
        self.__draw_p1()

    def __draw_p2(self):
        # draw signal of the selected channel
        if len(self.p2.items)>1:
            for item in self.p2.items[1:]:
                self.p2.removeItem(item)
        line = MultiLine(np.array([self.times]), np.array([self.analogs_pool[self.select_chn]['data']]), 'g')
        self.p2.addItem(line)
        
    def __draw_p1(self):
        # draw selected signals in p2
        self.p1.clear()
        line = MultiLine(np.array([self.times[self.selected_indexs_p2]]),np.array([self.analogs_pool[self.select_chn]['data'][self.selected_indexs_p2]]),'g')
        self.p1.addItem(line)

    def __band_pass(self):
        # dialog for inputting values used in band pass filtering
        w=QtWidgets.QInputDialog()
        w.setLabelText('Band Pass')
        w.textValueSelected.connect(self.__filter_bandpass)
        w.exec_()

    def __band_stop(self):
        # dialog for inputting values used in band stop filtering
        w=QtWidgets.QInputDialog()
        w.setLabelText('Band Stop')
        w.textValueSelected.connect(self.__filter_bandstop)
        w.exec_()

    def __reset_popup(self):
        # pop up an alert box for confirmming reset data
        reset_msg = QtWidgets.QMessageBox()
        reset_msg.setIcon(QtWidgets.QMessageBox.Information)
        reset_msg.setText("Are you sure to discard the changes you make?")

        reset_msg.setStandardButtons(QtWidgets.QMessageBox.Reset | QtWidgets.QMessageBox.Cancel)
        reset_msg.setDefaultButton(QtWidgets.QMessageBox.Reset)
        reset_msg.buttonClicked.connect(self.__reset)
        reset_msg.exec_()

    def __reset(self,reset_message):
        # clear up filtered data
        if reset_message.text() == 'Reset':
            self.analogs_pool = dict()
            self.p1.clear()
            self.p2.clear()
            self.select_chn = None
            for i,ch in enumerate(self.unique_chan):
                self.p3_chns_button[i].setPen('k')

    def __save_popup(self):
        # pop up an alert box for confirmming saving data 
        save_msg = QtWidgets.QMessageBox()
        save_msg.setIcon(QtWidgets.QMessageBox.Information)
        save_msg.setText('Are you sure to save the data ?')

        save_msg.setStandardButtons(QtWidgets.QMessageBox.Save |  QtWidgets.QMessageBox.Cancel)
        save_msg.setDefaultButton(QtWidgets.QMessageBox.Save)
        save_msg.buttonClicked.connect(self.__save_data)
        save_msg.exec_()

    def __save_data(self,save_message):
        if save_message.text() == 'Save':
            datasave = ftl.flatten_dict(self.analogs_pool)
            # save filtered data into the file
            with hp.File(self.file_name,'a') as f:
                for key,value in datasave.iteritems():
                    if 'analogs/'+key in f:
                        del f['analogs'][key]
                    f['analogs'][key] = value
                f.flush()
                print('Data has been saved')
    
    def __filter_bandpass(self,value):
        # check the legality of inputted bandpass value 
        values = value.split(',')
        if len(values)==2 and self.select_chn != None:
            if ftl.is_num(values[0]) and ftl.is_num(values[1]):
                val_0 = float(values[0])
                val_1 = float(values[1])
                if val_0<val_1:
                    sampling_rate = self.analogs_pool[self.select_chn]['sampling_rate']
                    # filtering data
                    result_data = ftl.band_pass(self.analogs_pool[self.select_chn]['data'],val_0,val_1,sampling_rate)
                    if result_data is not None:
                        self.analogs_pool[self.select_chn]['data'] = result_data
                        print('band pass applied to data')
                        self.__draw_p2()
                        self.__draw_p1()
        else:
            print('please check input value, or if special channel is selected')
            
    def __filter_bandstop(self,value):
        # check the legality of inputted bandstop value 
        values = value.split(',')
        if len(values)==2 and self.select_chn != None:
            if ftl.is_num(values[0]) and ftl.is_num(values[1]):
                val_0 = float(values[0])
                val_1 = float(values[1])
                if val_0<val_1:
                    sampling_rate = self.analogs_pool[self.select_chn]['sampling_rate']
                    # filtering data
                    result_data = ftl.band_stop(self.analogs_pool[self.select_chn]['data'],val_0,val_1,sampling_rate)
                    if result_data is not None:
                        self.analogs_pool[self.select_chn]['data'] = result_data
                        print('band pass applied to data')
                        self.__draw_p2()
                        self.__draw_p1()
        else:
            print('please check input value, or if special channel is selected')
    
    def __filter_bandpass_nogui(self,data,value,sampling_rate):
        # filtering data without showing a GUI window using bandpass method
        if len(value) == 2:
            if ftl.is_num(value[0]) and ftl.is_num(value[1]):
                val_0 = float(value[0])
                val_1 = float(value[1])
                if val_0<val_1:
                    result_data = ftl.band_pass(data,val_0,val_1,sampling_rate)
                    return result_data
                else:
                    raise ValueError('input bandpass value is invalid')
            else:
                raise ValueError('input bandpass value is invalid')
        else:
            raise ValueError('input bandpass value is invalid')

    def __filter_bandstop_nogui(self,data,value,sampling_rate):
        # filtering data without showing a GUI window using bandstop method
        if len(value) == 2:
            if ftl.is_num(value[0]) and ftl.is_num(value[1]):
                val_0 = float(value[0])
                val_1 = float(value[1])
                if val_0<val_1:
                    result_data = ftl.band_stop(data,val_0,val_1,sampling_rate)
                    return result_data
                else:
                    raise ValueError('input bandpass value is invalid')
            else:
                raise ValueError('input bandpass value is invalid')
        else:
            raise ValueError('input bandpass value is invalid')

# This class is used to draw multiple lines for the AnalogFilter class in a memory-efficient way
class MultiLine(pg.QtGui.QGraphicsPathItem):
    def __init__(self, x, y,color):
        """x and y are 2D arrays of shape (Nplots, Nsamples)"""
        connect = np.ones(x.shape, dtype=bool)
        # don't draw the segment between each trace
        connect[:,-1] = 0 
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        pg.QtGui.QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen(color))
        
    # override because QGraphicsPathItem.shape is too expensive.
    def shape(self): 
        return pg.QtGui.QGraphicsItem.shape(self)
    def boundingRect(self):
        return self.path.boundingRect()
