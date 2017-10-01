import sys
import numpy as np
import time
import math
from . import pyqtgraph as pg
from .pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .pyqtgraph import dockarea
from . import func_tools as ftl
import h5py as hp

class SpikeDetection():
    def __init__(self):
        # create main GUI window
        self.thresholdDefault = -4
        # self.tmpDefault = 0.9
        self.current_page = 0
        self.chnPageNum = 0
        # segment widgets init pos
        self.pk0_roi0_pos0 = [2,0]
        self.pk0_roi0_pos1 = [10,0]

        self.pk0_roi1_pos0 = [10,0]
        self.pk0_roi1_pos1 = [20,0]
        self.pk0_roi_pos = [0,0]

        app = QtGui.QApplication([])
        win = QtGui.QMainWindow()
        win.setFixedWidth(1200)
        win.setFixedHeight(700)

        self.area = dockarea.DockArea()
        win.setCentralWidget(self.area)
        win.setWindowTitle("spike detection")
        # create docks, place them into the window one at a time
        d1 = dockarea.Dock("analog signals",size=(600,700))
        d2 = dockarea.Dock("waveforms", size=(600,350))
        d3 = dockarea.Dock("control panel", size=(600,350))
        
        d1.setAcceptDrops(False)
        d2.setAcceptDrops(False)
        d3.setAcceptDrops(False)

        self.area.addDock(d1,"left")
        self.area.addDock(d2,"right",d1)
        self.area.addDock(d3,"bottom",d2)

        # add plot analog signals panes to d1
        self.row_num = 8
        self.row_wins = [0]*self.row_num
        for i in range(self.row_num):
            self.row_wins[i] = pg.PlotWidget()
            self.row_wins[i].setMouseEnabled(x=True,y=False)
            self.row_wins[i].hideAxis('bottom')
            self.row_wins[i].hideAxis('left')
        for ite in self.row_wins:
            d1.addWidget(ite)

        # add waveform pane to d2
        self.pk0 = pg.PlotWidget()
        self.pk0.hideAxis('bottom')
        self.pk0.hideAxis('left')
        self.pk0.setMouseEnabled(x=True,y=True)        
        d2.addWidget(self.pk0)

        # add control panel to d3
        self.pc = pg.LayoutWidget()
        d3.addWidget(self.pc)
        # channel list
        selChnLabel = QtGui.QLabel("select \nchannel:")
        selChnLabel.setFixedWidth(80)
        selChnLabel.setFixedHeight(30)
        self.pc.addWidget(selChnLabel,row=0,col=0)   
        self.chnList = QtGui.QComboBox()
        self.chnList.setFixedWidth(120)
        self.chnList.setFixedHeight(30)
        self.chnList.currentIndexChanged.connect(self.__selectChan)
        self.pc.addWidget(self.chnList,row=0,col=1,colspan=2)

        # use windows button
        useWdwLabel = QtGui.QLabel("use window in \nthis page:")
        useWdwLabel.setFixedWidth(100)
        useWdwLabel.setFixedHeight(30)
        self.pc.addWidget(useWdwLabel,row=0,col=3,colspan=2)           
        self.wdwCheck = QtGui.QRadioButton()
        self.wdwCheck.setFixedWidth(100)
        self.wdwCheck.clicked.connect(self.__updateWindows)
        self.pc.addWidget(self.wdwCheck,row=0,col=5)

        self.pageLabel = QtGui.QLabel("Page: {0}/{1}".format(self.current_page,self.chnPageNum))
        self.pageLabel.setFixedWidth(120)
        self.pageLabel.setFixedHeight(15)
        self.pc.addWidget(self.pageLabel,row=1,col=0,colspan=2)        
        self.peakBeforeLabel = QtGui.QLabel("PeakBefore")
        self.peakBeforeLabel.setFixedWidth(70)
        self.peakBeforeLabel.setFixedHeight(10)
        self.pc.addWidget(self.peakBeforeLabel,row=1,col=2)
        self.peakAfterLabel = QtGui.QLabel("PeakAfter")
        self.peakAfterLabel.setFixedWidth(70)
        self.peakAfterLabel.setFixedHeight(10)
        self.pc.addWidget(self.peakAfterLabel,row=1,col=3)

        # previous page button
        previousPageBtn = QtGui.QPushButton("Previous")
        # previousPageBtn.setMaximumWidth(80)
        previousPageBtn.setFixedWidth(80)
        previousPageBtn.clicked.connect(self.__previousPage)
        self.pc.addWidget(previousPageBtn,row=2,col=0)
        # next page button
        nextPageBtn = QtGui.QPushButton("Next")
        # nextPageBtn.setMaximumWidth(80)
        nextPageBtn.setFixedWidth(80)
        nextPageBtn.clicked.connect(self.__nextPage)
        self.pc.addWidget(nextPageBtn,row=2,col=1)

        # peak before button
        self.peakBeforeBtn = QtGui.QComboBox()
        self.peakBeforeBtn.setFixedWidth(70)
        self.peakBeforeBtn.addItems([str(ite) for ite in range(1,101)])
        self.peakBeforeBtn.setCurrentIndex(19)
        self.pc.addWidget(self.peakBeforeBtn,row=2,col=2)
        self.peakBeforeBtn.currentIndexChanged.connect(self.__draw_wave)

        # peak after button
        self.peakAfterBtn = QtGui.QComboBox()
        self.peakAfterBtn.setFixedWidth(70)
        self.peakAfterBtn.addItems([str(ite) for ite in range(1,101)])
        self.peakAfterBtn.setCurrentIndex(29)
        self.pc.addWidget(self.peakAfterBtn,row=2,col=3)
        self.peakAfterBtn.currentIndexChanged.connect(self.__draw_wave)

        # load file button
        loadBtn = QtGui.QPushButton("Load")
        # loadBtn.setMaximumWidth(70)  
        loadBtn.setFixedWidth(70)      
        loadBtn.clicked.connect(self.__load)
        self.pc.addWidget(loadBtn,row=2,col=4)

        # save result button
        saveBtn = QtGui.QPushButton("Save")
        # saveBtn.setMaximumWidth(70)
        saveBtn.setFixedWidth(70)
        saveBtn.clicked.connect(self.__save)
        self.pc.addWidget(saveBtn,row=2,col=5)

        # threshold label
        thresholdLabel = QtGui.QLabel()
        thresholdLabel.setText("Threshold: ")
        thresholdLabel.setFixedWidth(80)
        self.pc.addWidget(thresholdLabel,row=3,col=0) 
        thresholdUnitLabel = QtGui.QLabel()        
        thresholdUnitLabel.setText(" * sigma")
        self.pc.addWidget(thresholdUnitLabel,row=3,col=2)

        self.resetAllBtn = QtGui.QPushButton("use for all channels")
        self.resetAllBtn.setFixedWidth(150)
        self.resetAllBtn.clicked.connect(self.__resetAll)
        self.pc.addWidget(self.resetAllBtn,row=3,col=3,colspan=2)

        self.thresholdSB = QtGui.QPushButton()
        self.thresholdSB.setText(str(self.thresholdDefault))
        self.thresholdSB.setFixedWidth(70)
        self.thresholdSB.setStyleSheet("background-color: rgb(150,150,150);border-radius: 6px;")
        self.thresholdSB.clicked.connect(self.__resetThis)
        self.pc.addWidget(self.thresholdSB,row=3,col=1,colspan=1)

        self.autoThisBtn = QtGui.QPushButton("auto for this channel")
        self.autoThisBtn.setFixedWidth(150)
        self.autoThisBtn.clicked.connect(self.__autoThis)
        self.pc.addWidget(self.autoThisBtn,row=5,col=0,colspan=2)
        self.autoAllBtn = QtGui.QPushButton("auto for all channels")
        self.autoAllBtn.setFixedWidth(150)
        self.pc.addWidget(self.autoAllBtn,row=5,col=2,colspan=2)
        self.autoAllBtn.clicked.connect(self.__autoAll)

        # reset Layout button
        resetLayoutBtn = QtGui.QPushButton("Reset Layout")
        # resetLayoutBtn.setMaximumWidth(80)
        resetLayoutBtn.clicked.connect(self.__resetLayout)
        self.pc.addWidget(resetLayoutBtn,row=6,colspan=6)
        self.area_state = self.area.saveState()

        win.show()
        sys.exit(app.exec_())


    def __selectChan(self):
        self.__clearRows()
        if self.chnList.currentText():
            self.selectChan = self.chnList.currentText()
            self.__chnPageNum()
            self.current_page = 0
            self.__updatePage()
            self.__load_page_data()
            self.__redraw_threshold()
            self.__updateThrTim()
            self.__setWindowState()
    def __chnPageNum(self):
        with hp.File(self.file_name,"r") as f:
            data_nums = f["analogs"][self.selectChan]["data"].value.shape[0]
            sampling_rate = f["analogs"][self.selectChan]["sampling_rate"].value
            self.chnPageNum = int(math.floor(data_nums / (1.0*self.row_num*sampling_rate)))

    def __load_page_data(self):
        self.__clearRows()
        if hasattr(self,"selectChan"):
            with hp.File(self.file_name,"r") as f:
                sampling_rate = f["analogs"][self.selectChan]["sampling_rate"].value
                start_time = f["analogs"][self.selectChan]["start_time"].value
                start_point = sampling_rate*self.row_num*self.current_page
                end_point = sampling_rate*self.row_num*(self.current_page+1)
                self.page_data = f["analogs"][self.selectChan]["data"][start_point:end_point]
                self.sigma = np.median(np.abs(self.page_data)/0.6745)
                Thr = self.thresholds[self.selectChan] * self.sigma
            self.sampling_rate = sampling_rate
            self.row_wins_rois = [0]*self.row_num
            for i in range(self.row_num):
                start_point = i*sampling_rate
                end_point = (i+1)*sampling_rate
                if self.page_data[start_point:end_point].size:
                    ys = self.page_data[start_point:end_point]
                    xs = np.arange(ys.size)
                    line = MultiLine(np.array([xs]),np.array([ys]),"w")
                    self.row_wins[i].addItem(line)

                self.row_wins_rois[i] = pg.InfiniteLine(pos=Thr,angle=0,movable=False)
                self.row_wins_rois[i].setZValue(10)
                self.row_wins[i].addItem(self.row_wins_rois[i])

    def __updateWindows(self):
        if hasattr(self,"selectChan"):
            if self.wdwCheck.isChecked() is True:
                self.windowsState[self.selectChan + "_" + str(self.current_page)] = {"pos0":self.pk0_roi0.pos(),
                                                                                    "roi0_handle0":self.pk0_roi0.getLocalHandlePositions()[0][1],
                                                                                    "roi0_handle1":self.pk0_roi0.getLocalHandlePositions()[1][1],
                                                                                    "pos1":self.pk0_roi1.pos(),
                                                                                    "roi1_handle0":self.pk0_roi1.getLocalHandlePositions()[0][1],
                                                                                    "roi1_handle1":self.pk0_roi1.getLocalHandlePositions()[1][1]}

            elif self.wdwCheck.isChecked() is False:
                self.windowsState.pop(self.selectChan + "_" + str(self.current_page),None)

            self.__draw_wave()

    def __previousPage(self):
        if self.current_page >0:
            self.current_page -= 1
        elif self.current_page == 0:
            self.current_page = self.chnPageNum
        self.__setWindowState()
        self.__updatePage()
        self.__load_page_data()
        self.__redraw_threshold()

    def __nextPage(self):
        if self.current_page < self.chnPageNum:
            self.current_page += 1
        elif self.current_page == self.chnPageNum:
            self.current_page = 0
        self.__setWindowState()
        self.__updatePage()
        self.__load_page_data()
        self.__redraw_threshold()

    def __draw_wave(self):
        self.__clearWave()
        if hasattr(self,"selectChan"):
            Thr = self.thresholds[self.selectChan] * self.sigma
            peak_before = int(self.peakBeforeBtn.currentText())
            peak_after = int(self.peakAfterBtn.currentText())
            self.spike_peaks, self.spike_waveforms = self.__threshold_spike(self.page_data,Thr,peak_before,peak_after)

            detected_mask = self.__detect_now(self.spike_waveforms,self.selectChan,self.current_page)

            temp_time = np.arange(0,peak_before+peak_after)
            detected_spikes = self.spike_waveforms[detected_mask]
            not_detected_spikes = self.spike_waveforms[~detected_mask]
            detected_peaks = self.spike_peaks[detected_mask]
            if not_detected_spikes.size:
                times = np.vstack([temp_time for i in range(not_detected_spikes.shape[0])])
                self.not_detected_line = MultiLine(times, not_detected_spikes, 'w')
                self.pk0.addItem(self.not_detected_line)


            if detected_spikes.size:
                detected_peaks_amp = detected_spikes[:,peak_before]

                times = np.vstack([temp_time for i in range(detected_spikes.shape[0])])
                self.detected_line = MultiLine(times, detected_spikes, 'g')
                self.pk0.addItem(self.detected_line)

                self.peak_items = []
                for i in range(self.row_num):
                    start_idx = i*self.sampling_rate
                    end_idx = (i+1)*self.sampling_rate

                    mask_idx = (detected_peaks>=start_idx) & (detected_peaks<end_idx)

                    xs = detected_peaks[mask_idx] - start_idx
                    ys = detected_peaks_amp[mask_idx]
                    
                    self.peak_items.append(pg.ScatterPlotItem(symbol='o',brush='g',size=4,x=xs,y=ys))
                    self.row_wins[i].addItem(self.peak_items[i])

    def __clearWave(self):
        if hasattr(self,"detected_line"):
            self.pk0.removeItem(self.detected_line)
        if hasattr(self,"not_detected_line"):
            self.pk0.removeItem(self.not_detected_line)
        if hasattr(self,"peak_items"):
            for i in range(self.row_num):
                self.row_wins[i].removeItem(self.peak_items[i])

    def __threshold_spike(self,page_data,Thr,peak_before,peak_after):
        spike_peaks = self.__detect_spike_peak(page_data,Thr,peak_before,peak_after)
        # spike timestamp and waveform
        temp_waveform = []
        if spike_peaks.size:
            temp_idx = np.vstack([spike_peaks - peak_before, spike_peaks + peak_after]).T
            for idx in temp_idx:
                temp_waveform.append(page_data[idx[0]:idx[1]])
        spike_waveforms = np.array(temp_waveform)
        return spike_peaks,spike_waveforms
    def __detect_spike_peak(self,ang_data,Thr,peak_before,peak_after):
        if Thr < 0:
            dd_0 = np.where(ang_data<Thr)[0]
        elif Thr >=0:
            dd_0 = np.where(ang_data>=Thr)[0]
        dd_1 = np.diff(dd_0,n=1)
        dd_2 = np.where(dd_1 > 1)[0]+1
        dd_3 = np.split(dd_0,dd_2)
        spike_peak = []
        if Thr < 0:
            for ite in dd_3:
                if ite.size:
                    potent_peak = ite[ang_data[ite].argmin()]
                    if (potent_peak + peak_after <= ang_data.shape[0]) and (potent_peak - peak_before >= 0):
                        spike_peak.append(potent_peak)
        elif Thr >=0:
            for ite in dd_3:
                if ite.size:
                    potent_peak = ite[ang_data[ite].argmax()]
                    if (potent_peak + peak_after <= ang_data.shape[0]) and (potent_peak - peak_before >= 0):
                        spike_peak.append(potent_peak)
        return np.array(spike_peak)
    def __detect_now(self,spike_waveforms,selectChan,current_page):
        if selectChan+"_"+str(current_page) in self.windowsState:
            use_shape0 = self.__pk0_roi0_pos(selectChan,current_page)
            spk_in_line = np.apply_along_axis(self.__in_select_line,1,spike_waveforms,use_shape0[0],use_shape0[1])

            use_shape1 = self.__pk0_roi1_pos(selectChan,current_page)
            spk_in_line1 = np.apply_along_axis(self.__in_select_line,1,spike_waveforms,use_shape1[0],use_shape1[1])
            detected_mask = spk_in_line & spk_in_line1
        else:        
            detected_mask = np.ones(spike_waveforms.shape[0],dtype=bool)
        return detected_mask
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

    def __pk0_roi0_pos(self,selectChan,current_page):
        key = selectChan + "_" + str(current_page)
        pos = self.windowsState[key]["roi0_handle0"] + self.windowsState[key]["pos0"]
        pk0_roi0_pos0 = [pos[0],pos[1]]
        pos = self.windowsState[key]["roi0_handle1"] + self.windowsState[key]["pos0"]
        pk0_roi0_pos1 = [pos[0],pos[1]]
        return [pk0_roi0_pos0,pk0_roi0_pos1]
    def __pk0_roi1_pos(self,selectChan,current_page):
        key = selectChan + "_" + str(current_page)
        pos = self.windowsState[key]["roi1_handle0"] + self.windowsState[key]["pos1"]
        pk0_roi1_pos0 = [pos[0],pos[1]]
        pos = self.windowsState[key]["roi1_handle1"] + self.windowsState[key]["pos1"]
        pk0_roi1_pos1 = [pos[0],pos[1]]
        return [pk0_roi1_pos0,pk0_roi1_pos1]

    def __load(self):
        # Load a file-selection dialog
        name = QtGui.QFileDialog.getOpenFileName(filter="h5 (*.h5)")
        self.file_name = name[0]
        if len(self.file_name) > 0:
            if hasattr(self,'selectChan'):
                delattr(self,'selectChan')
            self.chnList.clear()
            self.thresholdSB.setText(str(self.thresholdDefault))
            self.ang_chns = []
            with hp.File(self.file_name,'r') as f:
                for key in f.keys():
                    if key == 'analogs':
                        for chn in f['analogs'].keys():
                            if chn.startswith('analog_'):
                                self.ang_chns.append(chn)

            if not self.ang_chns:
                raise ValueError("analogs entity is empty, please check the loaded file")

            self.__initThrTim()
            self.windowsState = dict()
            # this sentence will also auto load self.__selectChan
            self.chnList.addItems(self.ang_chns)
            self.__autoAll()
            # self.__updatePage()
            self.__setWindowState()
            if hasattr(self,"pk0_roi0"):
                self.pk0.removeItem(self.pk0_roi0)
                self.pk0.removeItem(self.pk0_roi1)
                delattr(self,"pk0_roi0")
                delattr(self,"pk0_roi1")
            self.pk0_roi0 = pg.LineSegmentROI([self.pk0_roi0_pos0,self.pk0_roi0_pos1],self.pk0_roi_pos,pen="r")
            self.pk0_roi1 = pg.LineSegmentROI([self.pk0_roi1_pos0,self.pk0_roi1_pos1],self.pk0_roi_pos,pen="r")
            self.pk0.addItem(self.pk0_roi0)
            self.pk0.addItem(self.pk0_roi1)
            # self.pk0_roi0 = pg.LineSegmentROI([self.windowsState[self.selectChan + "_" + str(self.current_page)]["roi0_handle0"],
            #                                         self.windowsState[self.selectChan + "_" + str(self.current_page)]["roi0_handle1"]],
            #                                         self.windowsState[self.selectChan + "_" + str(self.current_page)]["pos0"],pen="r")

    def __initThrTim(self):
        self.thresholds = {key:self.thresholdDefault for key in self.ang_chns}

    def __updatePage(self):
        self.pageLabel.setText("Page: {0}/{1}".format(self.current_page,self.chnPageNum))
        self.pageLabel.setFixedHeight(15)

    def __setWindowState(self):
        if hasattr(self,"selectChan"):
            if self.selectChan + "_" + str(self.current_page) in self.windowsState:
                self.wdwCheck.setChecked(True)
            else:
                self.wdwCheck.setChecked(False)

    def __save(self):
        peak_before = int(self.peakBeforeBtn.currentText())
        peak_after = int(self.peakAfterBtn.currentText())
        with hp.File(self.file_name,"a") as f:
            for selectChan in self.ang_chns:
                sampling_rate = f["analogs"][selectChan]["sampling_rate"].value
                start_time = f["analogs"][selectChan]["start_time"].value
                data_nums = f["analogs"][selectChan]["data"].value.shape[0]
                chnPageNum = int(math.floor(data_nums / (1.0*self.row_num*sampling_rate)))
                all_waveforms = []
                all_times = []
                for current_page in range(chnPageNum+1):
                    start_point = sampling_rate*self.row_num*current_page
                    end_point = sampling_rate*self.row_num*(current_page+1)
                    page_data = f["analogs"][selectChan]["data"][start_point:end_point]
                    sigma = np.median(np.abs(page_data)/0.6745)
                    Thr = self.thresholds[selectChan] * sigma
                    spike_peaks, spike_waveforms = self.__threshold_spike(page_data,Thr,peak_before,peak_after)
                    detected_mask = self.__detect_now(spike_waveforms,selectChan,current_page)

                    detected_spikes = spike_waveforms[detected_mask]
                    detected_peaks = spike_peaks[detected_mask]

                    if detected_spikes.size:
                        detected_peaks = detected_peaks + start_point
                        all_waveforms.append(detected_spikes)
                        all_times.append(detected_peaks)

                if all_times:
                    all_times = np.hstack(all_times)
                    all_times = all_times*1000.0/sampling_rate
                    all_waveforms = np.vstack(all_waveforms)

                if 'spikes' in f.keys():
                    for chn in f['spikes'].keys():
                        if chn.startswith(selectChan.replace('analog','spike')):
                            del f['spikes'][chn]
                    
                key_time = 'spikes/'+selectChan.replace('analog','spike')+'_0/times'
                key_wave = 'spikes/'+selectChan.replace('analog','spike')+'_0/waveforms'
                f[key_time] = all_times
                f[key_wave] = all_waveforms
                print(selectChan," saved.")

            f.flush()
            time.sleep(0.5)
            self.__finish_msg()
    def __finish_msg(self):
        finish_msg = QtWidgets.QMessageBox()
        finish_msg.setIcon(QtWidgets.QMessageBox.Information)
        finish_msg.setText("Save Finished !")
        
        finish_msg.addButton("OK",QtGui.QMessageBox.YesRole)
        # err_msg.addButton("onlySaveProcessedChns",QtGui.QMessageBox.YesRole)
        finish_msg.exec_()
        
    def __resetThis(self):
        if hasattr(self,"selectChan"):
            w=QtWidgets.QInputDialog()
            w.setLabelText('Set minimal temperature value.\n0 <= MinTemp < MaxTemp\nDefault is 0.')
            w.textValueSelected.connect(self.__resetThisValue)
            w.exec_()

    def __resetThisValue(self,value):
        if ftl.is_num(value) and (float(value)!=float(self.thresholdSB.text())):
            self.thresholdSB.setText(str(value))
            self.thresholds[self.selectChan] = float(value)
            self.__updateThrTim()
            self.__redraw_threshold()
        
    def __resetAll(self):
        if hasattr(self,"ang_chns"):
            self.thresholds = {key:float(self.thresholdSB.text()) for key in self.ang_chns}
            self.__updateThrTim()
            self.__redraw_threshold()

    def __autoThis(self):
        if hasattr(self,"selectChan"):
            self.thresholdSB.setText(str(self.thresholdDefault))
            self.thresholds[self.selectChan] = self.thresholdDefault
            self.__updateThrTim()
            self.__redraw_threshold()

    def __autoAll(self):
        if hasattr(self, "ang_chns"):
            self.thresholdSB.setText(str(self.thresholdDefault))
            self.__initThrTim()
            self.__updateThrTim()
            self.__redraw_threshold()
    def __redraw_threshold(self):
        if hasattr(self,"selectChan"):
            Thr = self.thresholds[self.selectChan] * self.sigma
            for i in range(self.row_num):
                self.row_wins_rois[i].setPos(Thr)        
            self.__draw_wave()

    def __updateThrTim(self):
        self.thresholdSB.setText(str(self.thresholds[self.selectChan]))
    def __resetLayout(self):
        self.area.restoreState(self.area_state)
    def __clearRows(self):
        for i in range(len(self.row_wins)):
            self.row_wins[i].clear()   
    
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