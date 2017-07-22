# -*- coding: utf-8 -*-
'''
Module for offline spike sorting.
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import func_tools as ftl
import sys


class SpikeSorting():
    '''
    Class for offline spike sorting. 
    Two methods are provided for spike sorting, window discriminator and principle components analysis.
    A line widget and a polygon widget are provided for window discriminator and PCA discriminator respectively.
    The graphic user interface design and data plotting tools are based on an open-source project, PyQtGraph.

    Args
        pca_3d (Boolean):
            If true, a 3D view for the first three principle components of spikes will be shown.
            Default: False
        reclaim_space (Boolean):
            If true, the storage space that the file occupied will be reallocated, which will realsease
            extra storage space.
            Default: False
    Returns
        -
    Examples
        >>> SpikeSorting(True)
            Open the offline spike sorting window and a 3D view for the first three principle components of spikes.
    '''
    def __init__(self,pca_3d=False,reclaim_space=False):
        """
        Initialize the SpikeSorting class.
        """
        self.reclaim_space = reclaim_space
        self.pca_3d = pca_3d
        # self.colors=['b','w','g','r','y']
        self.colors=[(0,0,1*255,255),(1*255,1*255,1*255,255),(0,1*255,0,255),(1*255,0,0,255),(1*255,1*255,0,255),
                     (1*255,0,1*255,255),(0,1*255,1*255,255),(0,0.5*255,0.5*255,255),(0.5*255,0,0.5*255,255),(0.5*255,0.5*255,0,255)]
        self.color =[(0,0,1,0.5),    (1,1,1,0.5),            (0,1,0,0.5),    (1,0,0,0.5),    (1,1,0,0.5),
                     (1,0,1,0.5),        (0,1,1,0.5),        (0,0.5,0.5,0.5),        (0.5,0,0.5,0.5),        (0.5,0.5,0,0.5)]

        # create the main graphic user interface window
        app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow()
        self.win.resize=(1200,700)
        # if pca_3d = True, create a 3D view window
        if pca_3d is True:
            self.win2 = gl.GLViewWidget()
            self.win2.resizeGL(500,500)
            self.win2.show()
            self.win2.setWindowTitle('3-D PCA')
        
        # four sub-windows: p1, p2, p3, p4
        # p1: draw selected spikes' waveforms
        # p2: draw various control buttons
        # p3: draw principle components of spikes
        # p4: draw timestamps of spikes
        p1 = self.win.addPlot(row=0,col=0)
        p1.setMaximumHeight(600)
        p1.setMaximumWidth(600)
        p1.hideAxis('bottom')
        p1.hideAxis('left')
        # p1.setYRange(-1000,600)

        p2 = self.win.addViewBox(row=0,col=1)
        p2.setAspectLocked(True)
        p2.setMaximumHeight(600)
        p2.setMaximumWidth(120)
        p2.setMouseEnabled(x=False,y=False)

        p3 = self.win.addPlot(row=0,col=2)
        p3.setMaximumHeight(600)
        p3.setMaximumWidth(600)
        p3.hideAxis('left')
        p3.hideAxis('bottom')

        self.win.nextRow()
        p4 = self.win.addPlot(colspan=3)
        p4.setMouseEnabled(x=True,y=False)
        p4.setMaximumHeight(100)
        p4.setMaximumWidth(1200)
        p4.hideAxis('left')

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        self.p2_y_zero = 0
        # set load_data button in p2
        load_offset = -60
        p2_load_button = pg.ScatterPlotItem(symbol='s')
        p2_load_button.setData([{'pos':[-20,load_offset+self.p2_y_zero],'pen':'g','size':25}])
        self.p2.addItem(p2_load_button)
        p2_load_text = pg.TextItem('load',color='w')
        p2_load_text.setPos(-20-12,load_offset+self.p2_y_zero-6)
        self.p2.addItem(p2_load_text)
        # set save_data button in p2
        
        self.p2_save_button = pg.ScatterPlotItem(symbol='s')
        self.p2_save_button.setData([{'pos':[0,load_offset+self.p2_y_zero],'pen':'g','size':25}])
        self.p2.addItem(self.p2_save_button)
        p2_save_text = pg.TextItem('save',color='w')
        p2_save_text.setPos(0-12,load_offset+self.p2_y_zero-6)
        p2.addItem(p2_save_text)
        
        # set reset_data button in p2
        # reset_data button will discard results from previous manipulation, and reload data from file
        self.p2_reset_button = pg.ScatterPlotItem(symbol='s')
        self.p2_reset_button.setData([{'pos':[20,load_offset+self.p2_y_zero],'pen':'g','size':25}])
        self.p2.addItem(self.p2_reset_button)
        p2_reset_text = pg.TextItem('reset',color='w')
        p2_reset_text.setPos(20-12,load_offset+self.p2_y_zero-6)
        self.p2.addItem(p2_reset_text)

        # set pca button in p2
        # pca button is used for selecting certain principle component
        pca_offset = -100
        self.pca_used_num = 4
        self.p2_pca_buttonX = [0]*self.pca_used_num
        self.p2_pca_buttonY = [0]*self.pca_used_num
        for i in range(self.pca_used_num):
            self.p2_pca_buttonX[i] = pg.ScatterPlotItem(symbol='s')
            self.p2_pca_buttonX[i].setData([{'pos':[-8+i*12,self.p2_y_zero+pca_offset],'size':15,'brush':self.colors[i]}])
            self.p2.addItem(self.p2_pca_buttonX[i])

            self.p2_pca_buttonY[i] = pg.ScatterPlotItem(symbol='s')
            self.p2_pca_buttonY[i].setData([{'pos':[-8+i*12,self.p2_y_zero+pca_offset-15],'size':15,'brush':self.colors[i]}])
            self.p2.addItem(self.p2_pca_buttonY[i])

        p2_pca_txt_1 = pg.TextItem('PCA',color='w')
        p2_pca_txt_1.setPos(-32,pca_offset+20)
        self.p2.addItem(p2_pca_txt_1)
        p2_pca_txt_2 = pg.TextItem('1',color='w')
        p2_pca_txt_2.setPos(-13,pca_offset+20)
        self.p2.addItem(p2_pca_txt_2)
        p2_pca_txt_3 = pg.TextItem('2',color='w')
        p2_pca_txt_3.setPos(-1,pca_offset+20)
        self.p2.addItem(p2_pca_txt_3)
        p2_pca_txt_4 = pg.TextItem('3',color='w')
        p2_pca_txt_4.setPos(11,pca_offset+20)
        self.p2.addItem(p2_pca_txt_4)
        p2_pca_txt_5 = pg.TextItem('4',color='w')
        p2_pca_txt_5.setPos(23,pca_offset+20)
        self.p2.addItem(p2_pca_txt_5)
        p2_pca_txt_6 = pg.TextItem('X',color='w')
        p2_pca_txt_6.setPos(-25,pca_offset+7)
        self.p2.addItem(p2_pca_txt_6)
        p2_pca_txt_7 = pg.TextItem('Y',color='w')
        p2_pca_txt_7.setPos(-25,pca_offset-8)
        self.p2.addItem(p2_pca_txt_7)

        # self.p2_delUnit_button = pg.ScatterPlotItem(symbol='s')
        # self.p2_delUnit_button.setData([{'pos':[-23,self.p2_y_zero+load_offset+20],'size':15}])
        # self.p2.addItem(self.p2_delUnit_button)
        # p2_delUnit_txt = pg.TextItem('del unit',color='w')
        # p2_delUnit_txt.setPos(-16,self.p2_y_zero+load_offset+28)
        # self.p2.addItem(p2_delUnit_txt)

        self.p2_addUnit_button = pg.ScatterPlotItem(symbol='s')
        self.p2_addUnit_button.setData([{'pos':[-23,self.p2_y_zero+load_offset+33],'size':15}])
        self.p2.addItem(self.p2_addUnit_button)
        p2_addUnit_txt = pg.TextItem('add unit',color='w')
        p2_addUnit_txt.setPos(-16,self.p2_y_zero+load_offset+41)
        self.p2.addItem(p2_addUnit_txt)

        self.p2_delSpk_button = pg.ScatterPlotItem(symbol='s')
        self.p2_delSpk_button.setData([{'pos':[-23,self.p2_y_zero+load_offset+18],'size':14}])
        self.p2.addItem(self.p2_delSpk_button)
        p2_delSpk_txt = pg.TextItem('del spike',color='w')
        p2_delSpk_txt.setPos(-16,self.p2_y_zero+load_offset+25)
        self.p2.addItem(p2_delSpk_txt)

        p2_load_button.sigClicked.connect(self.__load_data)
        # count items number in p2 without channel items added
        self.p2_items_num_without_chan = len(self.p2.addedItems)

        # execute the application
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    def __load_data(self):
        '''
        load data file, and extract spike information in the file
        '''
        # create the file selection dialogue, and only files with hdf5 format can be selected.
        name = QtGui.QFileDialog.getOpenFileName(filter="hdf5 (*.h5)")
        self.file_name = name[0]
        if len(self.file_name) >0:
            print(self.file_name)
            self.win.setWindowTitle(self.file_name.split('/')[-1])
            self.p1.clear()
            self.p3.clear()
            self.p4.clear()
            # set initial values for segment widget and polygon widget
            self.p4_roi_pos1 = 0
            self.p4_roi_pos2 = 10000

            # set initial value for pca button
            self.pca_x = 0
            self.pca_y = 1
            
            if hasattr(self,'select_chan'):
                delattr(self,'select_chan')

            # read spike information in the loaded file
            with hp.File(self.file_name,'r') as f:
                self.unique_chan = list()
                for key in f.keys():
                    if key == 'spikes':
                        volt_min = []
                        volt_max = []
                        for tem_key in f[key].keys():
                            self.unique_chan.append('_'.join(tem_key.split('_')[:2]))
                            volt_min.append(f[key][tem_key]['waveforms'].value.min())
                            volt_max.append(f[key][tem_key]['waveforms'].value.max())
                        volt_min = min(volt_min)/3.0
                        volt_max = max(volt_max)/10.0

                # self.p1_roi_pos1 = [5,volt_max] 
                # self.p1_roi_pos2 = [5,volt_min]
                self.p1_roi_pos1_init = [5,volt_max] 
                self.p1_roi_pos2_init = [5,volt_min]
                self.p1_roi1_pos1_init = [15,volt_max]
                self.p1_roi1_pos2_init = [15,volt_min]                
                # extract spike channels information in the file
                self.unique_chan = list(set(self.unique_chan))
                self.unique_chan = sorted(self.unique_chan)
                # add update func to save button 
                # not add it in __init__, make sure that only after data loaded, save button can be used
                
                if len(self.p2.addedItems)<=self.p2_items_num_without_chan:
                    self.p2_save_button.sigClicked.connect(self.__save_popup)
                    self.p2_reset_button.sigClicked.connect(self.__reset_popup)
                    self.p2_addUnit_button.sigClicked.connect(self.__add_unit_popup)
                    self.p2_delSpk_button.sigClicked.connect(self.__del_spike_popup)
                
                # clean chans, units, rects buttons in p2
                for item in self.p2.addedItems[self.p2_items_num_without_chan:]:
                    self.p2.removeItem(item)

                # set channel button in p2
                self.chan_offset = 150
                self.p2_chans_button = [0]*len(self.unique_chan)
                for i in range(len(self.unique_chan)):
                    self.p2_chans_button[i] = pg.ScatterPlotItem(symbol='s')
                    # self.p2_chans_button[i].setData([-0.55],[12+(len(self.unique_chan)-1*i)],size=16)
                    self.p2_chans_button[i].setData([{'pos':[-21,self.chan_offset-13*i],'size':15,'brush':'w'}])
                    # self.p2_chans_button[i].setBrush('w')
                    self.p2.addItem(self.p2_chans_button[i])
                p2_chans_texts = [0]*len(self.unique_chan)
                for i in range(len(self.unique_chan)):
                    text = self.unique_chan[i]
                    p2_chans_texts[i] = pg.TextItem(text,color='w')
                    p2_chans_texts[i].setPos(-16,self.chan_offset+9-13*i)
                    self.p2.addItem(p2_chans_texts[i])
                self.unit_offset = self.chan_offset - 13*len(self.unique_chan) - 15
                
                # add update func to p2_chans_button
                for i in range(len(self.p2_chans_button)):
                    self.p2_chans_button[i].sigClicked.connect(self.__update_chan(i))
                self.p2_init_items_num = len(self.p2.addedItems)
                # initiate a variable to storage the sorted spikes
                self.spikes_pool = dict()

        elif len(self.file_name) == 0:
            print('No file is selected')
            return None
    def __save_popup(self):
        # pop up an alert box for confirmming saving data 
        save_msg = QtWidgets.QMessageBox()
        save_msg.setIcon(QtWidgets.QMessageBox.Information)
        save_msg.setText('Are you sure to save the data?')

        save_msg.setStandardButtons(QtWidgets.QMessageBox.Save |  QtWidgets.QMessageBox.Cancel)
        save_msg.setDefaultButton(QtWidgets.QMessageBox.Save)
        save_msg.buttonClicked.connect(self.__save_data)
        save_msg.exec_()

    def __reset_popup(self):
        # pop up an alert box for confirmming reset data
        save_msg = QtWidgets.QMessageBox()
        save_msg.setIcon(QtWidgets.QMessageBox.Information)
        save_msg.setText('Are you sure to reset the data, and discard the change you make?')

        save_msg.setStandardButtons(QtWidgets.QMessageBox.Reset |  QtWidgets.QMessageBox.Cancel)
        save_msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
        save_msg.buttonClicked.connect(self.__reset_data)
        save_msg.exec_()
    def __reset_data(self,reset_message):
        if reset_message.text() == 'Reset':
            # clear up resorting data
            self.spikes_pool = dict()
            # reset chans button color
            for i in range(len(self.unique_chan)):
                self.p2_chans_button[i].setPen('w')
            # clear items in p1,p3,p4
            self.p1.clear()
            self.p3.clear()
            self.p4.clear()
            # clear units,rects buttons in p2
            for item in self.p2.addedItems[self.p2_init_items_num:]:
                self.p2.removeItem(item)
            # reset pca button color
            for i in range(self.pca_used_num):
                self.p2_pca_buttonX[i].setPen('w')
                self.p2_pca_buttonY[i].setPen('w')
            self.pca_x = 0
            self.pca_y = 1
            if self.pca_3d is True:
                for item in self.win2.items[:]:
                    self.win2.items.remove(item)

            if hasattr(self,'select_chan'):
                delattr(self,'select_chan')
        elif reset_message.text() == 'Cancel':
            pass
    def __save_data(self,save_message):
        if save_message.text()=='Save':
            datasave = dict()
            for chan in self.spikes_pool.keys():
                unit_unique = np.unique(self.spikes_pool[chan]['units'])
                unit_unique = np.array(unit_unique,dtype=np.int32)
                for unit in unit_unique:
                    key = chan+'_'+str(unit)
                    datasave[key] = dict()
                    mask = self.spikes_pool[chan]['units'] == unit
                    datasave[key]['times'] = self.spikes_pool[chan]['times'][mask]
                    datasave[key]['waveforms'] = self.spikes_pool[chan]['waveforms'][mask]
            datasave = ftl.flatten_dict(datasave)
            # save resorting data into the file
            with hp.File(self.file_name, 'a') as f:
                changed_chans = list(datasave.keys())
                changed_chans = list(set(['_'.join(ite.split('/')[0].split('_')[:2]) for ite in changed_chans]))
                for chan in changed_chans:
                    for key in f['spikes'].keys():
                        if key.startswith(chan):
                            del f['spikes'][key]
                for key,value in datasave.iteritems():                    
                    f['spikes'][key] = value
                f.flush()
                print('Data has beed resaved.')

        elif save_message.text()=='Cancel':
            pass
    def __del_spike_popup(self):
        del_spk_msg = QtWidgets.QMessageBox()
        del_spk_msg.setIcon(QtWidgets.QMessageBox.Information)
        if hasattr(self,'select_chan'):
            del_spk_msg.setText('Are you sure to delete spikes in unit -1 ?')
            del_spk_msg.setStandardButtons(QtWidgets.QMessageBox.Yes |  QtWidgets.QMessageBox.Cancel)
            del_spk_msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
        else:
            del_spk_msg.setText('please select a channel first!')
            del_spk_msg.setStandardButtons(QtWidgets.QMessageBox.Cancel)
            del_spk_msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        del_spk_msg.buttonClicked.connect(self.__del_spk)
        del_spk_msg.exec_()

    def __del_spk(self,del_msg):
        if del_msg.text() == '&Yes':
            retain_mask = self.spikes_pool[self.select_chan]['units'] != -1
            self.spikes_pool[self.select_chan]['units'] = self.spikes_pool[self.select_chan]['units'][retain_mask]
            self.spikes_pool[self.select_chan]['times'] = self.spikes_pool[self.select_chan]['times'][retain_mask]
            self.spikes_pool[self.select_chan]['waveforms'] = self.spikes_pool[self.select_chan]['waveforms'][retain_mask]
            self.spikes_pool[self.select_chan]['waveforms_range'] = self.spikes_pool[self.select_chan]['waveforms_range'][retain_mask]
            self.spikes_pool[self.select_chan]['waveforms_pca'] = self.spikes_pool[self.select_chan]['waveforms_pca'][retain_mask]

            self.waveforms_pca_max = np.abs(self.spikes_pool[self.select_chan]['waveforms_pca']).max()
            self.__update_roi_p4()
            self.__update_roi_p1()
            self.__update_roi1_p1()            
            self.__update_roi_p3()
            self.__draw_p1()
            self.__draw_p4()
            self.__draw_p3()
            if self.pca_3d is True:
                self.__draw_3D_PCA()

        elif del_msg.text() == 'Cancel':
            pass

    def __add_unit_popup(self):
        add_unit_msg = QtWidgets.QMessageBox()
        add_unit_msg.setIcon(QtWidgets.QMessageBox.Information)
        if hasattr(self,'select_chan'):
            add_unit_msg.setText('Are you sure to add unit {0} to this channel?'.format(str(self.unique_unit[-1]+1)))
            add_unit_msg.setStandardButtons(QtWidgets.QMessageBox.Yes |  QtWidgets.QMessageBox.Cancel)
            add_unit_msg.setDefaultButton(QtWidgets.QMessageBox.Yes)
        else:
            add_unit_msg.setText('please select a channel first!')
            add_unit_msg.setStandardButtons(QtWidgets.QMessageBox.Cancel)
            add_unit_msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        add_unit_msg.buttonClicked.connect(self.__add_unit)
        add_unit_msg.exec_()

    def __add_unit(self,msg):
        if msg.text() == '&Yes':
            self.unique_unit.append(self.unique_unit[-1]+1)
            # clear units,rects buttons in p2
            for item in self.p2.addedItems[self.p2_init_items_num:]:
                self.p2.removeItem(item)
            # add units button to p2
            # set unique_unit button in p2
            # [{'pos':[-25,self.chan_offset-13*i],'size':15,'brush':'w'}]
            self.p2_units_button = [0]*len(self.unique_unit)
            for i in range(len(self.unique_unit)):
                self.p2_units_button[i] = pg.ScatterPlotItem()
                self.p2_units_button[i].setData([{'pos':[-23,self.unit_offset-13*i],'size':15,'brush':self.colors[i]}])
                self.p2.addItem(self.p2_units_button[i])
            self.p2_units_rects = [0]*len(self.unique_unit)
            for i in range(len(self.unique_unit)):
                self.p2_units_rects[i] = pg.ScatterPlotItem(symbol='s')
                self.p2_units_rects[i].setData([{'pos':[18,self.unit_offset-13*i],'size':15,'brush':self.colors[i]}])
                self.p2.addItem(self.p2_units_rects[i])
            p2_units_texts = [0]*len(self.unique_unit)
            for i in range(len(self.unique_unit)):
                text = 'unit %s'%str(self.unique_unit[i])
                p2_units_texts[i] = pg.TextItem(text,color=self.colors[i])
                p2_units_texts[i].setPos(-18,self.unit_offset-13*i+8)
                self.p2.addItem(p2_units_texts[i])
            # add update func to p2_units_button,p2_units_rects
            for i,unit in enumerate(self.unique_unit):
                self.p2_units_button[i].sigClicked.connect(self.__update_unit_button(unit))
                self.p2_units_rects[i].sigClicked.connect(self.__update_unit_rect(unit))

        elif msg.text() == 'Cancel':
            pass

    def __update_chan(self,chan_id):
        # recording the spike channel used currently
        def click():
            self.select_chan = str(self.unique_chan[chan_id])
            self.__add_to_memory(self.select_chan)
            self.unique_unit = list(np.unique(self.spikes_pool[self.select_chan]['units']))
            if -1 not in self.unique_unit:
                self.unique_unit.insert(0,-1)
            print("%s is selected"%self.select_chan)
            for i in range(len(self.unique_chan)):
                if i== chan_id:
                    self.p2_chans_button[i].setPen('r')
                else:
                    self.p2_chans_button[i].setPen('w')
            
            # clear items in p1,p3,p4
            self.p1.clear()
            self.p3.clear()
            self.p4.clear()
            # add roi item and draw func to p1,p3,p4
            self.selected_indexs_p4 = np.array(np.where((self.spikes_pool[self.select_chan]['times']>self.p4_roi_pos1) & \
                                                        (self.spikes_pool[self.select_chan]['times']<self.p4_roi_pos2))[0],dtype=np.int32)
            self.p1_roi_pos1 = self.p1_roi_pos1_init
            self.p1_roi_pos2 = self.p1_roi_pos2_init

            self.p1_roi = pg.LineSegmentROI([self.p1_roi_pos1, self.p1_roi_pos2],pen='r')
            self.p1.addItem(self.p1_roi)
            self.p1_roi.setZValue(10)

            self.p1_roi1_pos1 = self.p1_roi1_pos1_init
            self.p1_roi1_pos2 = self.p1_roi1_pos2_init
            self.p1_roi1 = pg.LineSegmentROI([self.p1_roi1_pos1, self.p1_roi1_pos2],pen='r')
            self.p1.addItem(self.p1_roi1)
            self.p1_roi1.setZValue(10)

            self.waveforms_pca_max = np.abs(self.spikes_pool[self.select_chan]['waveforms_pca']).max()
            self.p3_roi_pos1 = [0,0]
            self.p3_roi_pos2 = [self.waveforms_pca_max/2.5,self.waveforms_pca_max/2.5]
            self.p3_roi_pos1_init = [0,0]
            self.p3_roi_pos2_init = [self.waveforms_pca_max/2.5,self.waveforms_pca_max/2.5]
            if self.pca_3d is True:
                self.win2.opts['distance'] = abs(self.waveforms_pca_max)*1.5
            
            self.p3_roi_pos_init = [[0,0],[0,self.waveforms_pca_max/2.5],[self.waveforms_pca_max/2.5,self.waveforms_pca_max/2.5],[self.waveforms_pca_max/2.5,0]]
            self.p3_roi = pg.PolyLineROI(self.p3_roi_pos_init, pen='r',closed=True)
            self.p3_roi.setZValue(10)
            self.p3.addItem(self.p3_roi)
            self.roi_p3_pos = np.array(self.p3_roi_pos_init) + np.array([self.p3_roi.pos()[0]+self.p3_roi.pos()[1]])
            
            self.p4_roi = pg.LinearRegionItem()
            self.p4_roi.setRegion([self.p4_roi_pos1,self.p4_roi_pos2])
            self.p4_roi.setZValue(10)
            self.p4.addItem(self.p4_roi,ignoreBounds=True)

            # set update func to self.p2_pca_buttonX, self.p2_pca_buttonY
            if len(self.p2.addedItems) == self.p2_init_items_num:
                self.p2_pca_buttonX[self.pca_x].setPen('r')
                self.p2_pca_buttonY[self.pca_y].setPen('r')
                for i in range(self.pca_used_num):
                    self.p2_pca_buttonX[i].sigClicked.connect(self.__update_pca_x(i))
                    self.p2_pca_buttonY[i].sigClicked.connect(self.__update_pca_y(i))
            # clear units,rects buttons in p2
            for item in self.p2.addedItems[self.p2_init_items_num:]:
                self.p2.removeItem(item)
            # add units button to p2
            # set unique_unit button in p2
            # [{'pos':[-25,self.chan_offset-13*i],'size':15,'brush':'w'}]
            self.p2_units_button = [0]*len(self.unique_unit)
            for i in range(len(self.unique_unit)):
                self.p2_units_button[i] = pg.ScatterPlotItem()
                self.p2_units_button[i].setData([{'pos':[-23,self.unit_offset-13*i],'size':15,'brush':self.colors[i]}])
                self.p2.addItem(self.p2_units_button[i])
            self.p2_units_rects = [0]*len(self.unique_unit)
            for i in range(len(self.unique_unit)):
                self.p2_units_rects[i] = pg.ScatterPlotItem(symbol='s')
                self.p2_units_rects[i].setData([{'pos':[18,self.unit_offset-13*i],'size':15,'brush':self.colors[i]}])
                self.p2.addItem(self.p2_units_rects[i])
            p2_units_texts = [0]*len(self.unique_unit)
            for i in range(len(self.unique_unit)):
                text = 'unit %s'%str(self.unique_unit[i])
                p2_units_texts[i] = pg.TextItem(text,color=self.colors[i])
                p2_units_texts[i].setPos(-18,self.unit_offset-13*i+8)
                self.p2.addItem(p2_units_texts[i])
            # set update func to p4_roi,p1_roi,p3_roi
            # self.p4_roi.sigRegionChangeFinished.connect(self.__update_roi_p4)
            self.p4_roi.sigRegionChanged.connect(self.__update_roi_p4)
            self.p1_roi.sigRegionChangeFinished.connect(self.__update_roi_p1)
            self.p1_roi1.sigRegionChangeFinished.connect(self.__update_roi1_p1)
            
            self.p3_roi.sigRegionChangeFinished.connect(self.__update_roi_p3)
            self.__draw_p4()
            self.__draw_p3()
            self.__draw_p1()
            if self.pca_3d is True:
                self.__draw_3D_PCA()
            # add update func to p2_units_button,p2_units_rects
            for i,unit in enumerate(self.unique_unit):
                self.p2_units_button[i].sigClicked.connect(self.__update_unit_button(unit))
                self.p2_units_rects[i].sigClicked.connect(self.__update_unit_rect(unit))
            
        return click

    def __add_to_memory(self, select_chan):
        # add data in new selected spike channel to variable self.spikes_pool
        if select_chan not in self.spikes_pool.keys():
            times = list()
            waveforms = list()
            units = list()
            with hp.File(self.file_name,'r') as f:
                for key in f['spikes'].keys():
                    if key.startswith(select_chan):
                        time = f['spikes'][key]['times'].value
                        waveform = f['spikes'][key]['waveforms'].value
                        unit = int(key.split('_')[-1])
                        unit = np.ones(time.shape,dtype=np.int32)*unit
                        times.append(time)
                        waveforms.append(waveform)
                        units.append(unit)
            times = np.hstack(times)
            units = np.hstack(units)
            waveforms = np.vstack(waveforms)
            # sort data
            sort_index = np.argsort(times)
            units = units[sort_index]
            waveforms = waveforms[sort_index]
            times = times[sort_index]
            self.spikes_pool[select_chan] = dict()
            self.spikes_pool[select_chan]['times'] = times
            self.spikes_pool[select_chan]['units'] = units
            self.spikes_pool[select_chan]['waveforms'] = waveforms

            # calculate waveform range
            waveforms_max = np.apply_along_axis(max,1,self.spikes_pool[select_chan]['waveforms'])
            waveforms_min = np.apply_along_axis(min,1,self.spikes_pool[select_chan]['waveforms'])
            self.spikes_pool[select_chan]['waveforms_range'] = np.vstack([waveforms_max,waveforms_min]).T
            # calculate waveform PCA
            scaler = StandardScaler()
            scaler.fit(self.spikes_pool[select_chan]['waveforms'])
            waveforms_scaled = scaler.transform(self.spikes_pool[select_chan]['waveforms'])
            pca = PCA(n_components=self.pca_used_num)
            pca.fit(waveforms_scaled)
            self.spikes_pool[select_chan]['waveforms_pca'] = pca.transform(waveforms_scaled)

    def __update_pca_x(self,x_id):
        # select special principle component to be show in x axis
        def update_x():
            self.pca_x = x_id
            self.__draw_p3()
            for i in range(self.pca_used_num):
                if i == self.pca_x:
                    self.p2_pca_buttonX[i].setPen('r')
                else:
                    self.p2_pca_buttonX[i].setPen('w')
        return update_x
    def __update_pca_y(self,y_id):
        # select special principle component to be show in y axis
        def update_y():
            self.pca_y = y_id
            self.__draw_p3()
            for i in range(self.pca_used_num):
                if i == self.pca_y:
                    self.p2_pca_buttonY[i].setPen('r')
                else:
                    self.p2_pca_buttonY[i].setPen('w')
        return update_y
    
    def __draw_p1(self):
        # draw waveforms of selected spikes in p4
        if len(self.p1.items)>2:
            for item in self.p1.items[2:]:
                self.p1.removeItem(item)
        temp_waveforms = self.spikes_pool[self.select_chan]['waveforms'][self.selected_indexs_p4]
        temp_units = self.spikes_pool[self.select_chan]['units'][self.selected_indexs_p4]
        for i,unit in enumerate(self.unique_unit):
            mask = temp_units == unit
            temp_waveform = temp_waveforms[mask]
            temp_time = np.ones(temp_waveform.shape,dtype=np.int32)*np.arange(temp_waveform.shape[1],dtype=np.int32)
            lines = MultiLine(temp_time,temp_waveform,self.colors[i])        
            self.p1.addItem(lines)

    def __draw_p3(self):
        # draw selected principle components of spikes
        if len(self.p3.items)>1:
            for item in self.p3.items[1:]:
                self.p3.removeItem(item)
        
        for i,unit in enumerate(self.unique_unit):
            mask = self.spikes_pool[self.select_chan]['units']==unit
            self.p3.scatterPlot(self.spikes_pool[self.select_chan]['waveforms_pca'][mask][:,self.pca_x],\
                                    self.spikes_pool[self.select_chan]['waveforms_pca'][mask][:,self.pca_y],\
                                    symbolBrush=self.colors[i],size=4)
    def __draw_p4(self):
        # draw timestamps of spikes in special recording channel
        if len(self.p4.items)>1:
            for item in self.p4.items[1:]:
                self.p4.removeItem(item)
        for i,unit in enumerate(self.unique_unit):
            temp_data = self.spikes_pool[self.select_chan]['waveforms_range'][self.spikes_pool[self.select_chan]['units']==unit]
            temp_time = self.spikes_pool[self.select_chan]['times'][self.spikes_pool[self.select_chan]['units']==unit]
            temp_time = np.ones(temp_data.shape,dtype=np.int32)*temp_time[:,np.newaxis]
            lines = MultiLine(temp_time,temp_data,self.colors[i])
            self.p4.addItem(lines)

    def __update_roi_p4(self):
        # update time bin selected widget in p4
        minX,maxX = self.p4_roi.getRegion()
        self.selected_indexs_p4 = np.where((self.spikes_pool[self.select_chan]['times']>minX) & (self.spikes_pool[self.select_chan]['times']<maxX))[0]
        self.selected_indexs_p4 = np.array(self.selected_indexs_p4,dtype=np.int32)
        self.__draw_p1()
    def __update_roi_p1(self):
        # update segment widget in p1
        pos = self.p1_roi.getLocalHandlePositions()[0][1]+self.p1_roi.pos()
        self.p1_roi_pos1 = [pos[0],pos[1]]
        pos = self.p1_roi.getLocalHandlePositions()[1][1]+self.p1_roi.pos()
        self.p1_roi_pos2 = [pos[0],pos[1]]

    def __update_roi1_p1(self):
        # update segment roi1 widget in p1
        pos = self.p1_roi1.getLocalHandlePositions()[0][1]+self.p1_roi1.pos()
        self.p1_roi1_pos1 = [pos[0],pos[1]]
        pos = self.p1_roi1.getLocalHandlePositions()[1][1]+self.p1_roi1.pos()
        self.p1_roi1_pos2 = [pos[0],pos[1]]

    def __update_roi_p3(self):
        # update polygon widget in p3        
        pos = []
        for hand in self.p3_roi.getHandles():
            tem_pos = hand.pos()+self.p3_roi.pos()
            pos.append([tem_pos[0],tem_pos[1]])
        self.roi_p3_pos = np.array(pos)
        

    def __indexs_select_p1(self):
        # get indexs of selected spikes in p1
        temp_waveforms = self.spikes_pool[self.select_chan]['waveforms'][self.selected_indexs_p4]
        spk_in_line = np.apply_along_axis(self.__in_select_line,1,temp_waveforms,self.p1_roi_pos1,self.p1_roi_pos2)
        changed_index = np.where(spk_in_line==True)[0]
        changed_index = np.array(changed_index,dtype=np.int32)

        spk_in_line1 = np.apply_along_axis(self.__in_select_line,1,temp_waveforms,self.p1_roi1_pos1,self.p1_roi1_pos2)
        changed_index1 = np.where(spk_in_line1==True)[0]
        changed_index1 = np.array(changed_index1,dtype=np.int32)

        changed_index =  np.intersect1d(changed_index, changed_index1)
        return changed_index + self.selected_indexs_p4[0]

    # check whether a spike's waveform is intersect with segment widget
    def __in_select_line(self,temp_spike,pos_1,pos_2):
        pos_3y = temp_spike[:-1]
        pos_3x = np.ones(pos_3y.shape,dtype=np.int32)*np.arange(pos_3y.shape[0])
        pos_4y = temp_spike[1:]
        pos_4x = np.ones(pos_3y.shape,dtype=np.int32)*np.arange(1,pos_3y.shape[0]+1)
        pos_3_4 = np.vstack([ pos_3x,pos_3y,pos_4x,pos_4y]).T
        is_insect = np.apply_along_axis(self.__intersect,1,pos_3_4,pos_1,pos_2)
        return np.any(is_insect)
    def __ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    def __intersect(self,pos34,C,D):
        A = pos34[:2]
        B = pos34[2:4]
        return self.__ccw(A,C,D) != self.__ccw(B,C,D) and self.__ccw(A,B,C) != self.__ccw(A,B,D)
    def __update_unit_button(self,change_to_unit):
        # resorting selected spikes in p1
        def click():
            self.selected_indexs_p1 = self.__indexs_select_p1()
            self.spikes_pool[self.select_chan]['units'][self.selected_indexs_p1] =  np.ones(self.selected_indexs_p1.shape)*change_to_unit
            self.__draw_p1()
            self.__draw_p4()
            self.__draw_p3()
            if self.pca_3d is True:
                self.__draw_3D_PCA()
        return click
    def __update_unit_rect(self,change_to_unit):
        # resorting selected spikes in p3
        def click():
            self.selected_indexs_p3 = self.__indexs_select_p3()
            self.spikes_pool[self.select_chan]['units'][self.selected_indexs_p3] = np.ones(self.selected_indexs_p3.shape)*change_to_unit
            self.__draw_p1()
            self.__draw_p4()
            self.__draw_p3()
            if self.pca_3d is True:
                self.__draw_3D_PCA()
        return click
    def __indexs_select_p3(self):
        # get indexs of selected spikes in p3
        pca_pos = self.spikes_pool[self.select_chan]['waveforms_pca']
        x_min = self.roi_p3_pos[:,0].min()
        x_max = self.roi_p3_pos[:,0].max()
        y_min = self.roi_p3_pos[:,1].min()
        y_max = self.roi_p3_pos[:,1].max()
        
        x = np.logical_and(self.spikes_pool[self.select_chan]['waveforms_pca'][:,self.pca_x]>x_min, \
                            self.spikes_pool[self.select_chan]['waveforms_pca'][:,self.pca_x]<x_max)
        y = np.logical_and(self.spikes_pool[self.select_chan]['waveforms_pca'][:,self.pca_y]>y_min, \
                            self.spikes_pool[self.select_chan]['waveforms_pca'][:,self.pca_y]<y_max)
        ind_0 = np.logical_and(x, y)
        ind_0 = np.where(ind_0 == True)[0]
        ind_0 = np.array(ind_0,dtype=np.int32)
        
        if ind_0.shape[0]>0:
            segments = []
            for i in range(self.roi_p3_pos.shape[0]-1):
                segments.append([self.roi_p3_pos[i],self.roi_p3_pos[i+1]])
            segments.append([self.roi_p3_pos[-1],self.roi_p3_pos[0]])
            segments = np.array(segments)
            temp_pcas = self.spikes_pool[self.select_chan]['waveforms_pca'][ind_0]
            temp_pcas = temp_pcas[:,[self.pca_x,self.pca_y]]
            
            is_intersect = np.apply_along_axis(self.__intersect_roi3,1,temp_pcas,segments)
            
            return ind_0[is_intersect]
        else:
            return np.array([],dtype=np.int32)
    # check whether a spike's principle components is in the polygon widget
    def __intersect_roi3(self,temp_point,segments):
        point_B_xs_min = segments[:,:,0].min(axis=1)
        point_B_xs_max = segments[:,:,0].max(axis=1)
        point_B_ys_min = segments[:,:,1].min(axis=1)
        point_B_ys_max = segments[:,:,1].max(axis=1)
        
        # ids_0 = point_B_xs_max > temp_point[self.pca_x]
        # ids_1 = point_B_ys_min < temp_point[self.pca_y]
        # ids_2 = point_B_ys_max > temp_point[self.pca_y]
        ids_0 = point_B_xs_max > temp_point[0]
        ids_1 = point_B_ys_min < temp_point[1]
        ids_2 = point_B_ys_max > temp_point[1]
        
        segms = segments[(ids_0 & ids_1 & ids_2)]
        tem_insect = []
        for seg in segms:
            A = temp_point
            # B = [seg[:,0].max(),temp_point[self.pca_y]]
            B = [seg[:,0].max(),temp_point[1]]
            
            C = seg[0]
            D = seg[1]
            if temp_point[self.pca_x]<seg[:,0].min():
                tem_insect.append(True)
            elif self.__ccw(A,C,D) != self.__ccw(B,C,D) and self.__ccw(A,B,C) != self.__ccw(A,B,D):
                tem_insect.append(True)
        if sum(tem_insect)%2==1:
            return True
        else:
            return False

    def __draw_3D_PCA(self):
        # do not use self.win2.items, but use self.win2.items[:], otherwise item can't be deleted
        for item in self.win2.items[:]:
            self.win2.items.remove(item)

        pos = self.spikes_pool[self.select_chan]['waveforms_pca'][:,:3]
        for i,unit in enumerate(self.unique_unit):
            sp1 = gl.GLScatterPlotItem(pos=pos[self.spikes_pool[self.select_chan]['units']==unit], size=self.waveforms_pca_max/150.0, color=self.color[i], pxMode=False)
            self.win2.addItem(sp1)

# The SpikeSorting class use this class to draw multiple lines quickly in a memory-efficient way.
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
