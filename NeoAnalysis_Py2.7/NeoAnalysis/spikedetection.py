import sys
import numpy as np
import math
from . import pyqtgraph as pg
from .pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .pyqtgraph import opengl as gl
from . import func_tools as ftl
import h5py as hp

class SpikeDetection():
    def __init__(self):
        # create the main GUI window
        app = QtGui.QApplication([])
        win = pg.GraphicsWindow(border=True,size=(1200,700))
        # win.resize=(1400,900)
        self.colors=[(0,0,1*255,255),(1*255,1*255,1*255,255),(0,1*255,0,255),(1*255,0,0,255),(1*255,1*255,0,255),
                    (1*255,0,1*255,255),(0,1*255,1*255,255),(0,0.5*255,0.5*255,255),(0.5*255,0,0.5*255,255),(0.5*255,0.5*255,0,255)]
        self.peak_before = 20
        self.peak_after = 30
        self.yrange_0 = -8000
        self.yrange_1 = 5000
        self.threshold = self.yrange_0/2.0
        height_1 = 100
        width_1 = 600
        width_2 = 800
        self.row_num = 8

        self.row_wins = [0]*self.row_num
        for i in range(self.row_num):
            self.row_wins[i] = win.addPlot(row=i,col=0)
            self.row_wins[i].setMaximumHeight(height_1)
            self.row_wins[i].setMaximumWidth(width_1)
            self.row_wins[i].setMouseEnabled(x=True,y=False)
            self.row_wins[i].hideAxis('bottom')
            self.row_wins[i].hideAxis('left')
            self.row_wins[i].setYRange(self.yrange_0,self.yrange_1,padding=0)
        self.pk0 = win.addPlot(row=0,col=1,rowspan=5)
        self.pk0.setMaximumHeight(height_1*5)
        self.pk0.setMaximumWidth(width_2)
        self.pk0.hideAxis('bottom')
        self.pk0.hideAxis('left')
        self.pk0.setMouseEnabled(x=True,y=True)
        self.pc = win.addViewBox(row=5,col=1,rowspan=3)
        self.pc.setMaximumHeight(height_1*3)
        self.pc.setMaximumWidth(width_2)
        self.pc.setAspectLocked(True)
        self.pc.setMouseEnabled(x=False,y=False)

        pc_load_button = pg.ScatterPlotItem(symbol='s')
        pc_load_button.setData([{'pos':[0,16],'pen':'g','size':20}])
        self.pc.addItem(pc_load_button)
        pc_load_text = pg.TextItem('load',color='w')
        pc_load_text.setPos(-1,15.5)
        self.pc.addItem(pc_load_text)
        pc_load_button.sigClicked.connect(self.__load_data)
        self.is_linked = False

        self.pc_save_button = pg.ScatterPlotItem(symbol='s')
        self.pc_save_button.setData([{'pos':[0,8.5],'pen':'g','size':20}])
        self.pc.addItem(self.pc_save_button)
        pc_save_text = pg.TextItem('save',color='w')
        pc_save_text.setPos(-1,8)
        self.pc.addItem(pc_save_text)

        self.pc_prev_button = pg.ScatterPlotItem(symbol='t3')
        self.pc_prev_button.setData([{'pos':[-19,16],'pen':'g','size':20}])
        self.pc.addItem(self.pc_prev_button)
        pc_prev_text = pg.TextItem('prev',color='w')
        pc_prev_text.setPos(-20,15.4)
        self.pc.addItem(pc_prev_text)

        self.pc_next_button = pg.ScatterPlotItem(symbol='t2')
        self.pc_next_button.setData([{'pos':[-17,16],'pen':'g','size':20}])
        self.pc.addItem(self.pc_next_button)
        pc_next_text = pg.TextItem('next',color='w')
        pc_next_text.setPos(-18,15.4)
        self.pc.addItem(pc_next_text)

        self.threshold_text = pg.TextItem('threshold:',color='w')
        self.threshold_text.setPos(-20,12)
        self.pc.addItem(self.threshold_text)
        select_row_x = -12
        select_row_xs = 1.5        

        self.yrange_text = pg.TextItem('YRange:[{0}, {1}]'.format(str(self.yrange_0),str(self.yrange_1)),color='w')
        self.yrange_text.setPos(-20,11)
        self.pc.addItem(self.yrange_text)
        self.pc_autoY_button = pg.ScatterPlotItem(symbol='s')
        self.pc_autoY_button.setData([{'pos':[-18.5,8.5],'pen':'g','size':20}])
        self.pc.addItem(self.pc_autoY_button)
        pc_adjustY_text = pg.TextItem('AutoY',color='w')
        pc_adjustY_text.setPos(-20,8)
        self.pc.addItem(pc_adjustY_text)
        
        self.pc_setY_button = pg.ScatterPlotItem(symbol='s')
        self.pc_setY_button.setData([{'pos':[-16,8.5],'pen':'g','size':20}])
        self.pc.addItem(self.pc_setY_button)
        pc_setY_text = pg.TextItem('SetY',color='w')
        pc_setY_text.setPos(-17,8)
        self.pc.addItem(pc_setY_text)
        self.time_text_0 = pg.TextItem('Time:',color='w')
        self.time_text_0.setPos(-20,14)
        self.pc.addItem(self.time_text_0)
        self.time_text_1 = pg.TextItem('Point:',color='w')
        self.time_text_1.setPos(-20,13)
        self.pc.addItem(self.time_text_1)

        self.page_text = pg.TextItem('page: {0} / {1}'.format(str(0),str(0)),color='w')
        self.page_text.setPos(-20,18)
        self.pc.addItem(self.page_text)        

        self.used_text = pg.TextItem('set: {0}'.format('no'),color='w')
        self.used_text.setPos(-8,18)
        self.pc.addItem(self.used_text)

        self.pc_detect_button = pg.ScatterPlotItem(symbol='s')
        self.pc_detect_button.setData([{'pos':[-11,16],'pen':'g','size':20}])
        self.pc.addItem(self.pc_detect_button)
        pc_detect_text = pg.TextItem('detect spike',color='w')
        pc_detect_text.setPos(-13,15.4)
        self.pc.addItem(pc_detect_text)

        self.pc_use_button = pg.ScatterPlotItem(symbol='s')
        self.pc_use_button.setData([{'pos':[-5,16],'pen':'g','size':20}])
        self.pc.addItem(self.pc_use_button)
        pc_use_text = pg.TextItem('use this',color='w')
        pc_use_text.setPos(-6.5,15.4)
        self.pc.addItem(pc_use_text)

        self.pc_peakBeforeM_button = pg.ScatterPlotItem(symbol='t3')
        self.pc_peakBeforeM_button.setData([{'pos':[-12,8.5],'pen':'g','size':20}])
        self.pc.addItem(self.pc_peakBeforeM_button)
        self.pc_peakBeforeA_button = pg.ScatterPlotItem(symbol='t2')
        self.pc_peakBeforeA_button.setData([{'pos':[-10,8.5],'pen':'g','size':20}])
        self.pc.addItem(self.pc_peakBeforeA_button)
        
        pc_peakBefore_text_0 = pg.TextItem('peak before',color='w')
        pc_peakBefore_text_0.setPos(-13,8)
        self.pc.addItem(pc_peakBefore_text_0)
        self.pc_peakBefore_text = pg.TextItem('{0}'.format(str(self.peak_before)),color='w')
        self.pc_peakBefore_text.setPos(-11.5,10)
        self.pc.addItem(self.pc_peakBefore_text)

        self.pc_peakAfterM_button = pg.ScatterPlotItem(symbol='t3')
        self.pc_peakAfterM_button.setData([{'pos':[-6,8.5],'pen':'g','size':20}])
        self.pc.addItem(self.pc_peakAfterM_button)
        self.pc_peakAfterA_button = pg.ScatterPlotItem(symbol='t2')
        self.pc_peakAfterA_button.setData([{'pos':[-4,8.5],'pen':'g','size':20}])
        self.pc.addItem(self.pc_peakAfterA_button)
        
        pc_peakAfter_text_0 = pg.TextItem('peak after',color='w')
        pc_peakAfter_text_0.setPos(-7,8)
        self.pc.addItem(pc_peakAfter_text_0)
        self.pc_peakAfter_text = pg.TextItem('{0}'.format(str(self.peak_after)),color='w')
        self.pc_peakAfter_text.setPos(-5.5,10)
        self.pc.addItem(self.pc_peakAfter_text)

        self.pc_initItem_num = len(self.pc.addedItems)

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


    def __load_data(self):
        # Load a file-selection dialog
        name = QtGui.QFileDialog.getOpenFileName(filter="h5 (*.h5)")
        self.file_name = name[0]
        if len(self.file_name) > 0:
            if hasattr(self,'select_chn'):
                delattr(self,'select_chn')
            for i in range(len(self.row_wins)):
                self.row_wins[i].clear()
            self.pk0.clear()
            self.row_wins_rois = [0]*self.row_num
            for i in range(self.row_num):
                self.row_wins_rois[i] = pg.InfiniteLine(pos=self.threshold,angle=0,movable=False,name=str(i))
                self.row_wins_rois[i].setZValue(10)
                self.row_wins[i].addItem(self.row_wins_rois[i])
                self.row_wins_rois[i].sigPositionChangeFinished.connect(self.__update_threshold(i))

            for item in self.pc.addedItems[self.pc_initItem_num:]:
                self.pc.removeItem(item)

            self.ang_chns = []
            with hp.File(self.file_name,'r') as f:
                for key in f.keys():
                    if key == 'analogs':
                        for chn in f['analogs'].keys():
                            if chn.startswith('analog_'):
                                self.ang_chns.append(chn)
            # set analog channel buttons in pc window
            self.pc_chn_title = pg.TextItem('channel',color='w')
            self.pc_chn_title.setPos(3,19.5)
            self.pc.addItem(self.pc_chn_title)
            self.pc_chns_button = [0]*len(self.ang_chns)
            self.pc_chns_text = [0]*len(self.ang_chns)
            for i, ch in enumerate(self.ang_chns):
                self.pc_chns_button[i] = pg.ScatterPlotItem(symbol='o')
                self.pc_chns_button[i].setData([{'pos':[4,17.7-i*1.3],'pen':'w','size':15}])
                self.pc.addItem(self.pc_chns_button[i])
                self.pc_chns_text[i] = pg.TextItem(ch.replace('analog_',''),color='w')
                self.pc_chns_text[i].setPos(5,18.3-i*1.3)
                self.pc.addItem(self.pc_chns_text[i])
            for i,ch in enumerate(self.ang_chns):
                self.pc_chns_button[i].sigClicked.connect(self.__update_channel(ch))
            if self.is_linked is False:
                self.pc_save_button.sigClicked.connect(self.__save_pop_0)
                self.pc_next_button.sigClicked.connect(self.__update_page_next)
                self.pc_prev_button.sigClicked.connect(self.__update_page_prev)
                self.pc_detect_button.sigClicked.connect(self.__detect_now)
                self.pc_use_button.sigClicked.connect(self.__use_this)
                self.pc_autoY_button.sigClicked.connect(self.__autoY)
                self.pc_setY_button.sigClicked.connect(self.__update_setY)
                self.pc_peakBeforeM_button.sigClicked.connect(self.__update_peakBeforeM)
                self.pc_peakBeforeA_button.sigClicked.connect(self.__update_peakBeforeA)
                self.pc_peakAfterM_button.sigClicked.connect(self.__update_peakAfterM)
                self.pc_peakAfterA_button.sigClicked.connect(self.__update_peakAfterA)
                self.is_linked = True

            self.pk0_roi0_pos0 = [0,0]
            self.pk0_roi0_pos1 = [10,0]
            self.pk0_roi1_pos0 = [10,0]
            self.pk0_roi1_pos1 = [20,0]

            self.pk0_roi_0 = pg.LineSegmentROI([self.pk0_roi0_pos0,self.pk0_roi0_pos1],pen='r')
            self.pk0.addItem(self.pk0_roi_0)
            self.pk0_roi_0.setZValue(10)
            self.pk0_roi_1 = pg.LineSegmentROI([self.pk0_roi1_pos0,self.pk0_roi1_pos1],pen='r')
            self.pk0.addItem(self.pk0_roi_1)
            self.pk0_roi_1.setZValue(10)

    def __update_threshold(self,row_i):
        def click():
            temp_threshold = self.row_wins_rois[row_i].getPos()[1]
            if self.yrange_0<temp_threshold<self.yrange_1:
                self.threshold = temp_threshold
                for i in range(self.row_num):
                    self.row_wins_rois[i].setPos(self.threshold)
            else:
                for i in range(self.row_num):
                    self.row_wins_rois[i].setPos(self.threshold)
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()

        return click
    def __update_channel(self,select_chn):
        def click():
            self.temp_chn = select_chn
            if not hasattr(self,'select_chn'):
                self.select_chn = self.temp_chn
                for i,chn in enumerate(self.ang_chns):
                    if chn == self.select_chn:
                        self.pc_chns_button[i].setBrush('r')
                    else:
                        self.pc_chns_button[i].setBrush((100,100,150))
                with hp.File(self.file_name,'r') as f:
                    self.start_time = f['analogs'][self.select_chn]['start_time'].value
                    self.sampling_rate = f['analogs'][self.select_chn]['sampling_rate'].value
                    self.sampling_rate = int(self.sampling_rate)
                    self.rows = np.arange(int(math.ceil(f['analogs'][self.select_chn]['data'].value.shape[0]*1.0/self.sampling_rate)))
                for i in range(self.row_num):
                    self.row_wins_rois[i].setMovable(True)
                self.threshold_shape = dict()
                self.row_page = np.floor_divide(self.rows,self.row_num)
                self.page_num =self.row_page.max()
                self.current_page = 0
                self.__load_page_data()
                self.__draw_rows()
                self.__autoY()
                self.__threshold_spike()
                self.__draw_pk0()
                self.__update_indicator()

            else:
                reset_msg = QtWidgets.QMessageBox()  
                reset_msg.setIcon(QtWidgets.QMessageBox.Information)    
                reset_msg.setText("Are you sure to discard the spikes detected in {0} without saving it?".format(self.select_chn))

                reset_msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
                reset_msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
                reset_msg.buttonClicked.connect(self.__reset_chn)
                reset_msg.exec_()

        return click
    def __reset_chn(self,reset_message):
        if reset_message.text() == '&Yes':
            self.select_chn = self.temp_chn
            for i,chn in enumerate(self.ang_chns):
                if chn == self.select_chn:
                    self.pc_chns_button[i].setBrush('r')
                else:
                    self.pc_chns_button[i].setBrush((100,100,150))
            with hp.File(self.file_name,'r') as f:
                self.start_time = f['analogs'][self.select_chn]['start_time'].value
                self.sampling_rate = f['analogs'][self.select_chn]['sampling_rate'].value
                self.rows = np.arange(int(math.ceil(f['analogs'][self.select_chn]['data'].value.shape[0]*1.0/self.sampling_rate)))
            for i in range(self.row_num):
                self.row_wins_rois[i].setMovable(True)
            self.threshold_shape = dict()
            self.row_page = np.floor_divide(self.rows,self.row_num)
            self.page_num =self.row_page.max()
            self.current_page = 0
            self.__load_page_data()
            self.__draw_rows()
            self.__autoY()
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()
    def __update_page_next(self):
        if hasattr(self,'select_chn'):
            if self.current_page < self.page_num:
                self.current_page += 1
            elif self.current_page == self.page_num:
                self.current_page = 0
            self.__load_page_data()
            self.__draw_rows()
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()
    def __update_page_prev(self):
        if hasattr(self,'select_chn'):
            if self.current_page >0:
                self.current_page -= 1
            elif self.current_page == 0:
                self.current_page = self.page_num
            self.__load_page_data()
            self.__draw_rows()
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()

    def __clear_peak_pots(self):
        for i in range(self.row_num):
            if len(self.row_wins[i].items)>2:
                for item in self.row_wins[i].items[2:]:
                    self.row_wins[i].removeItem(item)
    def __use_this(self):
        if hasattr(self,'select_chn'):
            self.threshold_shape[self.current_page] = {'threshold':self.threshold,'shape0':self.use_shape0,'shape1':self.use_shape1}
            self.__update_indicator()
    def __detect_now(self):
        if hasattr(self,'select_chn'):
            if len(self.pk0.items)>2:
                for item in self.pk0.items[2:]:
                    self.pk0.removeItem(item)

            self.__clear_peak_pots()
            waveforms = []
            rows_in_detect = []
            peaks_time = []
            for i,ite in enumerate(self.spike_waveforms):
                if ite.shape[0]>0:
                    rows_in_detect.append(np.ones(ite.shape[0],dtype=np.int32)*i)
                    peaks_time.append(self.page_time[i][self.spike_peaks[i]])
                    waveforms.append(ite)

            if waveforms:
                waveforms = np.vstack(waveforms)
                rows_in_detect = np.hstack(rows_in_detect)
                peaks_time = np.hstack(peaks_time)
                peaks = waveforms[:,self.peak_before]
                
                self.use_shape0 = self.__pk0_roi0_pos()
                spk_in_line = np.apply_along_axis(self.__in_select_line,1,waveforms,self.use_shape0[0],self.use_shape0[1])

                self.use_shape1 = self.__pk0_roi1_pos()
                spk_in_line1 = np.apply_along_axis(self.__in_select_line,1,waveforms,self.use_shape1[0],self.use_shape1[1])
                selected_mask = spk_in_line & spk_in_line1

                temp_time = np.arange(0,self.peak_before+self.peak_after)
                selected_wave = waveforms[selected_mask]
                noselect_wave = waveforms[~selected_mask]

                selected_rows_in_detect = rows_in_detect[selected_mask]
                selected_peaks_time = peaks_time[selected_mask]
                selected_peaks = peaks[selected_mask]
                unique_row = np.unique(selected_rows_in_detect)
                for id_row in unique_row:
                    row_mask = selected_rows_in_detect == id_row
                    item = pg.ScatterPlotItem(symbol='o',brush='g',size=4,x=selected_peaks_time[row_mask],y=selected_peaks[row_mask])
                    self.row_wins[id_row].addItem(item)

                if selected_wave.size:
                    times = np.vstack([temp_time for i in range(selected_wave.shape[0])])
                    line = MultiLine(times, selected_wave, 'g')
                    self.pk0.addItem(line)
                if noselect_wave.size:
                    times = np.vstack([temp_time for i in range(noselect_wave.shape[0])])
                    line1 = MultiLine(times, noselect_wave, 'w')
                    self.pk0.addItem(line1)

                # selected_waveforms = waveforms[]
                # print changed_index
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

    def __pk0_roi0_pos(self):
        pos = self.pk0_roi_0.getLocalHandlePositions()[0][1]+self.pk0_roi_0.pos()
        pk0_roi0_pos0 = [pos[0],pos[1]]
        pos = self.pk0_roi_0.getLocalHandlePositions()[1][1]+self.pk0_roi_0.pos()
        pk0_roi0_pos1 = [pos[0],pos[1]]
        return [pk0_roi0_pos0,pk0_roi0_pos1]
    def __pk0_roi1_pos(self):
        pos = self.pk0_roi_1.getLocalHandlePositions()[0][1]+self.pk0_roi_1.pos()
        pk0_roi1_pos0 = [pos[0],pos[1]]
        pos = self.pk0_roi_1.getLocalHandlePositions()[1][1]+self.pk0_roi_1.pos()
        pk0_roi1_pos1 = [pos[0],pos[1]]
        return [pk0_roi1_pos0,pk0_roi1_pos1]
    def __load_page_data(self):
        self.select_rows = self.rows[self.row_page==self.current_page]
        self.page_data = []
        self.page_time = []
        with hp.File(self.file_name,'r') as f:
            for ite in self.select_rows:
                idx_0 = ite*self.sampling_rate
                idx_1 = idx_0 + self.sampling_rate
                temp_data = f['analogs'][self.select_chn]['data'][idx_0:idx_1]
                temp_time = np.arange(idx_0,idx_1)
                temp_time = temp_time[:temp_data.shape[0]]
                self.page_data.append(temp_data)
                self.page_time.append(temp_time)   

    def __draw_rows(self):
        for i in range(self.row_num):
            if len(self.row_wins[i].items)>1:
                for item in self.row_wins[i].items[1:]:
                    self.row_wins[i].removeItem(item)
        
        for i in range(len(self.select_rows)):
            line = MultiLine(np.array([self.page_time[i]]), np.array([self.page_data[i]]), 'w')
            self.row_wins[i].addItem(line)
    def __update_indicator(self):
        self.page_text.setText('page: {0} / {1}'.format(str(self.current_page),str(self.page_num)),color='w')
        self.threshold_text.setText('threshold: {0}'.format(self.threshold),color='w')
        self.yrange_text.setText('YRange:[{0}, {1}]'.format(str(self.yrange_0),str(self.yrange_1)),color='w')
        self.pc_peakBefore_text.setText('{0}'.format(str(self.peak_before)))
        self.pc_peakAfter_text.setText('{0}'.format(str(self.peak_after)))
        # self.time_text_0.setText('Time:[{0}, {1}]'.format(),color='w')
        point_0 = self.page_time[0][0]
        point_1 = self.page_time[-1][-1]
        time_0 = point_0*1000.0/self.sampling_rate+self.start_time
        time_1 = point_1*1000.0/self.sampling_rate+self.start_time
        
        self.time_text_1.setText('Point:[{0}, {1}]'.format(str(point_0),str(point_1)))
        self.time_text_0.setText('Time:[{0}, {1}] ms'.format(str(time_0),str(time_1)))
        
        if self.current_page in self.threshold_shape.keys():
            self.used_text.setText('set: {0}'.format('yes'),color='w')
        else:
            self.used_text.setText('set: {0}'.format('no'),color='w')
    def __autoY(self):
        if hasattr(self,'select_chn'):
            self.yrange_0 = min([ite.min() for ite in self.page_data])
            self.yrange_1 = max([ite.max() for ite in self.page_data])
            self.threshold = self.yrange_0/2.0
            for i in range(self.row_num):
                self.row_wins[i].setYRange(self.yrange_0,self.yrange_1,padding=0)
                self.row_wins_rois[i].setPos(self.threshold)
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()
    def __update_setY(self):
        # dialog for inputting YRange values
        if hasattr(self,'select_chn'):
            w=QtWidgets.QInputDialog()
            w.setLabelText('Band Stop')
            w.textValueSelected.connect(self.__setY)
            w.exec_()
    def __setY(self,value):
        # check the legality of inputted bandstop value 
        values = value.split(',')
        if len(values)==2:
            if ftl.is_num(values[0]) and ftl.is_num(values[1]):
                val_0 = float(values[0])
                val_1 = float(values[1])
                if val_0<val_1:
                    self.yrange_0 = val_0
                    self.yrange_1 = val_1
                    self.threshold = self.yrange_0/2.0
                    for i in range(self.row_num):
                        self.row_wins[i].setYRange(self.yrange_0,self.yrange_1,padding=0)
                        self.row_wins_rois[i].setPos(self.threshold)
                    self.__threshold_spike()
                    self.__draw_pk0()
                    self.__update_indicator()

    def __threshold_spike(self):
        self.__clear_peak_pots()
        self.spike_peaks = []
        for i in range(len(self.page_time)):
            self.spike_peaks.append(self.__detect_spike_peak(self.page_data[i]))
        # spike timestamp and waveform
        self.spike_waveforms = []
        for i,temp_peak in enumerate(self.spike_peaks):
            if temp_peak.size:
                temp_idx = np.vstack([temp_peak - self.peak_before, temp_peak + self.peak_after]).T
                temp_waveform = []
                for idx in temp_idx:
                    temp_waveform.append(self.page_data[i][idx[0]:idx[1]])
                temp_waveform = np.array(temp_waveform)
                self.spike_waveforms.append(temp_waveform)
            else:
                self.spike_waveforms.append(np.array([]))

    def __detect_spike_peak(self,ang_data):
        if self.threshold < 0:
            dd_0 = np.where(ang_data<self.threshold)[0]
        elif self.threshold >=0:
            dd_0 = np.where(ang_data>=self.threshold)[0]
        dd_1 = np.diff(dd_0,n=1)
        dd_2 = np.where(dd_1 > 1)[0]+1
        dd_3 = np.split(dd_0,dd_2)
        spike_peak = []
        if self.threshold < 0:
            for ite in dd_3:
                if ite.size:
                    potent_peak = ite[ang_data[ite].argmin()]
                    if (potent_peak + self.peak_after <= ang_data.shape[0]) and (potent_peak - self.peak_before >= 0):
                        spike_peak.append(potent_peak)
        elif self.threshold >=0:
            for ite in dd_3:
                if ite.size:
                    potent_peak = ite[ang_data[ite].argmax()]
                    if (potent_peak + self.peak_after <= ang_data.shape[0]) and (potent_peak - self.peak_before >= 0):
                        spike_peak.append(potent_peak)
        return np.array(spike_peak)

    def __draw_pk0(self):
        # draw signal of the selected channel
        if len(self.pk0.items)>2:
            for item in self.pk0.items[2:]:
                self.pk0.removeItem(item)

        temp_time = np.arange(0,self.peak_before+self.peak_after)
        for temp_waveform in self.spike_waveforms:
            if temp_waveform.size:
                times = np.vstack([temp_time for i in range(temp_waveform.shape[0])])
                line = MultiLine(times, temp_waveform, 'w')
                self.pk0.addItem(line)
    def __update_peakBeforeM(self):
        if hasattr(self,'select_chn'):
            if self.peak_before>1:
                self.peak_before -= 1
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()

    def __update_peakBeforeA(self):
        if hasattr(self,'select_chn'):
            if self.peak_before<100:
                self.peak_before += 1
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()

    def __update_peakAfterM(self):
        if hasattr(self,'select_chn'):
            if self.peak_after>1:
                self.peak_after -= 1
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()
    def __update_peakAfterA(self):
        if hasattr(self,'select_chn'):
            if self.peak_after<100:
                self.peak_after += 1
            self.__threshold_spike()
            self.__draw_pk0()
            self.__update_indicator()

    def __save_pop_0(self):
        if hasattr(self,'select_chn'):
            chan_exist = False
            with hp.File(self.file_name,'a') as f:
                if 'spikes' in f:
                    for key in f['spikes'].keys():
                        if key.startswith(self.select_chn.replace('analog','spike')):
                            chan_exist = True
                            break
            if chan_exist is True:
                save_msg = QtWidgets.QMessageBox()
                save_msg.setIcon(QtWidgets.QMessageBox.Information)
                save_msg.setText('spike_{0} already exists. Do you want to rewrite it ?'.format(self.select_chn.replace('analog_','')))

                save_msg.setStandardButtons(QtWidgets.QMessageBox.Save |  QtWidgets.QMessageBox.Cancel)
                save_msg.setDefaultButton(QtWidgets.QMessageBox.Save)
                save_msg.buttonClicked.connect(self.__save_popup_1)
                save_msg.exec_()
            else:
                false_pages = np.setdiff1d(np.unique(self.row_page),self.threshold_shape.keys(),True)
                if false_pages.shape[0]>0:
                    save_msg_1 = QtWidgets.QMessageBox()
                    save_msg_1.setIcon(QtWidgets.QMessageBox.Information)
                    save_msg_1.setText('Thresholds in These pages are not set. Do you want to ignore data in these pages?  {0}'.format(str(false_pages.tolist())))
                    save_msg_1.setStandardButtons(QtWidgets.QMessageBox.Save |  QtWidgets.QMessageBox.Cancel)
                    save_msg_1.setDefaultButton(QtWidgets.QMessageBox.Save)
                    save_msg_1.buttonClicked.connect(self.__save_data_1)
                    save_msg_1.exec_()
                else:
                    self.__save_data()
    def __save_popup_1(self,save_msg):
        if save_msg.text() == 'Cancel':
            print('Cancel')
            pass
        elif save_msg.text() == 'Save':
            false_pages = np.setdiff1d(np.unique(self.row_page),self.threshold_shape.keys(),True)
            if false_pages.shape[0]>0:
                save_msg_1 = QtWidgets.QMessageBox()
                save_msg_1.setIcon(QtWidgets.QMessageBox.Information)
                save_msg_1.setText('Thresholds in These pages are not set. Do you want to ignore data in these pages?  {0}'.format(str(false_pages.tolist())))
                save_msg_1.setStandardButtons(QtWidgets.QMessageBox.Save |  QtWidgets.QMessageBox.Cancel)
                save_msg_1.setDefaultButton(QtWidgets.QMessageBox.Save)
                save_msg_1.buttonClicked.connect(self.__save_data_1)
                save_msg_1.exec_()
    def __save_data_1(self,save_msg):
        if save_msg.text() == 'Save':
            self.__save_data()
        elif save_msg.text() == 'Cancel':
            print 'cancel'
            pass
    def __save_data(self):
        with hp.File(self.file_name,'a') as f:
            all_waveforms = []
            all_times = []
            for current_page, threshold_shape in self.threshold_shape.iteritems():
                select_rows = self.rows[self.row_page==current_page]
                threshold = threshold_shape['threshold']
                shape0 = threshold_shape['shape0']
                shape1 = threshold_shape['shape1']
                page_data = []
                page_time = []
                for ite in select_rows:
                    idx_0 = ite*self.sampling_rate
                    idx_1 = idx_0 + self.sampling_rate
                    temp_data = f['analogs'][self.select_chn]['data'][idx_0:idx_1]
                    temp_time = np.arange(idx_0,idx_1)
                    temp_time = temp_time[:temp_data.shape[0]]
                    page_data.append(temp_data)
                    page_time.append(temp_time)
                spike_peaks = []
                for i in range(len(page_time)):
                    spike_peaks.append(self.__result_detect_spike_peak(page_data[i],threshold))
                spike_waveforms = []
                for i,temp_peak in enumerate(spike_peaks):
                    if temp_peak.size:
                        temp_idx = np.vstack([temp_peak - self.peak_before, temp_peak + self.peak_after]).T
                        temp_waveform = []
                        for idx in temp_idx:
                            temp_waveform.append(page_data[i][idx[0]:idx[1]])
                        temp_waveform = np.array(temp_waveform)
                        spike_waveforms.append(temp_waveform)
                    else:
                        spike_waveforms.append(np.array([]))
                waveforms = []
                rows_in_detect = []
                peaks_time = []
                for i,ite in enumerate(spike_waveforms):
                    if ite.shape[0]>0:
                        rows_in_detect.append(np.ones(ite.shape[0],dtype=np.int32)*i)
                        peaks_time.append(page_time[i][spike_peaks[i]])
                        waveforms.append(ite)
                if waveforms:
                    waveforms = np.vstack(waveforms)
                    rows_in_detect = np.hstack(rows_in_detect)
                    peaks_time = np.hstack(peaks_time)
                    peaks = waveforms[:,self.peak_before]
                    use_shape0 = shape0
                    spk_in_line = np.apply_along_axis(self.__in_select_line,1,waveforms,use_shape0[0],use_shape0[1])

                    use_shape1 = shape1
                    spk_in_line1 = np.apply_along_axis(self.__in_select_line,1,waveforms,use_shape1[0],use_shape1[1])
                    selected_mask = spk_in_line & spk_in_line1

                    selected_wave = waveforms[selected_mask]
                    selected_peaks_time = peaks_time[selected_mask]

                    all_waveforms.append(selected_wave)
                    all_times.append(selected_peaks_time)
            if all_times:
                all_waveforms = np.vstack(all_waveforms)
                all_times = np.hstack(all_times)
                if 'spikes' in f.keys():
                    for chn in f['spikes'].keys():
                        if chn.startswith(self.select_chn.replace('analog','spike')):
                            del f['spikes'][chn]
                    
                key_time = 'spikes/'+self.select_chn.replace('analog','spike')+'_0/times'
                key_wave = 'spikes/'+self.select_chn.replace('analog','spike')+'_0/waveforms'
                f[key_time] = all_times*1000.0/self.sampling_rate + self.start_time
                f[key_wave] = all_waveforms
                f.flush()
                print('data are saved now.')
    def __result_detect_spike_peak(self,ang_data,threshold):
        if threshold < 0:
            dd_0 = np.where(ang_data<threshold)[0]
        elif threshold >=0:
            dd_0 = np.where(ang_data>=threshold)[0]
        dd_1 = np.diff(dd_0,n=1)
        dd_2 = np.where(dd_1 > 1)[0]+1
        dd_3 = np.split(dd_0,dd_2)
        spike_peak = []
        if threshold < 0:
            for ite in dd_3:
                if ite.size:
                    potent_peak = ite[ang_data[ite].argmin()]
                    if (potent_peak + self.peak_after <= ang_data.shape[0]) and (potent_peak - self.peak_before >= 0):
                        spike_peak.append(potent_peak)
        elif threshold >=0:
            for ite in dd_3:
                if ite.size:
                    potent_peak = ite[ang_data[ite].argmax()]
                    if (potent_peak + self.peak_after <= ang_data.shape[0]) and (potent_peak - self.peak_before >= 0):
                        spike_peak.append(potent_peak)
        return np.array(spike_peak)

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
