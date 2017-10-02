'''
Module for offline spike sorting.
This work is based on:
    * Bo Zhang, Ji Dai - first version
'''
import copy
from . import pyqtgraph as pg
from .pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .pyqtgraph import Qt
from .pyqtgraph import dockarea
import numpy as np
from math import floor,ceil
import time
import shutil
import subprocess
from . import func_tools as ftl
import platform
import os
import h5py as hp
import quantities as pq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pywt
from scipy.special import erfc
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
        self.pca_used_num = 3
        self.sortAlgorithms = ["Wavelets&SPC"]
        self.PCAused = ["1-2","1-3","2-3"]    
        self.unitsNum = range(1,10)
        self.unitsNum = [str(ite) for ite in self.unitsNum]    
        alpha=190
        self.colors=[(1*255,1*255,1*255,alpha),(0,1*255,0,alpha),(1*255,0,0,alpha),(1*255,1*255,0,alpha),(0,0,1*255,alpha),
                     (1*255,0,1*255,alpha),(0,1*255,1*255,alpha),(0,0.5*255,0.5*255,alpha),(0.5*255,0,0.5*255,alpha),(0.5*255,0.5*255,0,alpha)]
        self.color =[(1,1,1,0.5),            (0,1,0,0.5),    (1,0,0,0.5),    (1,1,0,0.5),       (0,0,1,0.5),
                     (1,0,1,0.5),        (0,1,1,0.5),        (0,0.5,0.5,0.5),        (0.5,0,0.5,0.5),        (0.5,0.5,0,0.5)]

        self.pk0_roi0_pos0 = [0,0]
        self.pk0_roi0_pos1 = [10,0]
        self.pk0_roi1_pos0 = [10,0]
        self.pk0_roi1_pos1 = [20,0]
        
        app = QtGui.QApplication([])
        self.win = QtGui.QMainWindow()
        self.win.setFixedWidth(1200)
        self.win.setFixedHeight(700) #700

        # if pac_3d is True, create a 3D view window
        if self.pca_3d is True:
            from .pyqtgraph import opengl as gl
            self.gl = gl
            self.win2 = self.gl.GLViewWidget()
            self.win2.resizeGL(500,500)
            self.win2.show()
            self.win2.setWindowTitle('3-D PCA')

        self.area = dockarea.DockArea()
        self.win.setCentralWidget(self.area)
        self.win.setWindowTitle("spike sorting")
        # create docks, place them into the window one at a time
        d0 = dockarea.Dock("waveforms",size=(500,600))  
        d1 = dockarea.Dock("control panel", size=(200,600))
        d2 = dockarea.Dock("PCA", size=(500,600))
        d3 = dockarea.Dock("timestamps",size=(1200,100))
        
        d0.setAcceptDrops(False)
        d1.setAcceptDrops(False)
        d2.setAcceptDrops(False)
        d3.setAcceptDrops(False)

        self.area.addDock(d0,"left")
        self.area.addDock(d3,"bottom",d0)
        self.area.addDock(d1,"right",d0)
        self.area.addDock(d2,"right",d1)
        # add waveform pane to d0
        self.pk0 = pg.PlotWidget()
        self.pk0.hideAxis('bottom')
        self.pk0.hideAxis('left')
        self.pk0.setMouseEnabled(x=True,y=True)        
        d0.addWidget(self.pk0)

        # add waveform pane to d2
        self.pk2 = pg.PlotWidget()
        self.pk2.hideAxis('bottom')
        self.pk2.hideAxis('left')
        self.pk2.setMouseEnabled(x=True,y=True)        
        d2.addWidget(self.pk2)

        # add waveform pane to d3
        self.pk3 = pg.PlotWidget()
        self.pk3.hideAxis('bottom')
        self.pk3.hideAxis('left')
        self.pk3.setMouseEnabled(x=True,y=False)        
        d3.addWidget(self.pk3)


        # add control panel to d1
        self.pc = pg.LayoutWidget()
        d1.addWidget(self.pc)
        
        # channel list
        self.chnLabel = QtGui.QLabel()
        self.chnLabel.setText("Channel:")
        self.chnLabel.setFixedWidth(70)
        self.pc.addWidget(self.chnLabel,row=0,col=0,colspan=3) 
        
        self.chnList = QtGui.QComboBox()
        self.chnList.setFixedWidth(70)
        # self.chnList.setFixedHeight(30)
        self.chnList.currentIndexChanged.connect(self.__selectChan)
        self.pc.addWidget(self.chnList,row=0,col=3,colspan=3)

        # save this channel
        saveThisChannel = QtGui.QLabel()
        saveThisChannel.setText("SaveThisChannel:")
        saveThisChannel.setFixedWidth(140)
        self.pc.addWidget(saveThisChannel,row=1,col=0,colspan=5)
        # self.saveChannelCheck = QtGui.QCheckBox()
        self.saveChannelCheck = QtGui.QCheckBox()
        self.saveChannelCheck.setFixedWidth(20)
        self.saveChannelCheck.clicked.connect(self.__saveThisChannelCheck)
        self.pc.addWidget(self.saveChannelCheck,row=1,col=5,colspan=1)

        # auto sorting this channel
        autoSortThisLabel = QtGui.QLabel()
        autoSortThisLabel.setText("AutoSortThisChannel:")
        autoSortThisLabel.setFixedWidth(140)
        self.pc.addWidget(autoSortThisLabel,row=2,col=0,colspan=5)
        # self.autoSortThisCheck = QtGui.QRadioButton()
        self.autoSortThisCheck = QtGui.QCheckBox()
        self.autoSortThisCheck.setFixedWidth(20)
        self.autoSortThisCheck.clicked.connect(self.__autoSortThisCheck)
        self.pc.addWidget(self.autoSortThisCheck,row=2,col=5,colspan=1)

        # algorithms args
        self.algorithmArgTree = pg.TreeWidget()
        self.algorithmArgTree.setColumnCount(2)
        self.algorithmArgTree.setHeaderLabels(["param","value"])        
        self.pc.addWidget(self.algorithmArgTree,row=4,col=0,colspan=6)
        self.algorithmArgTree.clear()

        # algorithms list
        self.algorithmList = QtGui.QComboBox()
        self.algorithmList.setFixedWidth(140)
        self.algorithmList.addItems(self.sortAlgorithms)
        self.algorithmList.currentIndexChanged.connect(self.__useThisAlgorithm)
        self.__useThisAlgorithm()
        self.pc.addWidget(self.algorithmList,row=3,col=0,colspan=6)

        
        # manual sort
        manualShapeLabel = QtGui.QLabel()
        manualShapeLabel.setText("ManuShape:")
        manualShapeLabel.setFixedWidth(80)
        self.pc.addWidget(manualShapeLabel,row=5,col=0,colspan=3)
        manualPCALabel = QtGui.QLabel()
        manualPCALabel.setText("ManuPCA:")
        manualPCALabel.setFixedWidth(80)
        self.pc.addWidget(manualPCALabel,row=5,col=3,colspan=3)
        # manual sort, shape
        self.shapeSortBtns = []
        for i,color in enumerate(self.colors):
            self.shapeSortBtns.append(QtGui.QPushButton())
            self.shapeSortBtns[i].setText(str(i))
            self.shapeSortBtns[i].setFixedWidth(18)
            self.shapeSortBtns[i].setFixedHeight(18)
            self.shapeSortBtns[i].setStyleSheet("background-color: rgb{0};border-radius: 6px;".format(color[:3]))
            self.shapeSortBtns[i].pressed.connect(self.__autoShapePress(i))
            self.shapeSortBtns[i].released.connect(self.__autoShapeRelease(i))    
            self.pc.addWidget(self.shapeSortBtns[i],row=6+i%5,col=int(i/5))
            
        # manual sort, PCA
        self.PCASortBtns = []
        for i,color in enumerate(self.colors):
            self.PCASortBtns.append(QtGui.QPushButton())
            self.PCASortBtns[i].setText(str(i))
            self.PCASortBtns[i].setFixedWidth(18)
            self.PCASortBtns[i].setFixedHeight(18)
            self.PCASortBtns[i].pressed.connect(self.__autoPCAPress(i))
            self.PCASortBtns[i].released.connect(self.__autoPCARelease(i))     
            self.PCASortBtns[i].setStyleSheet("background-color: rgb{0};border-radius: 6px;".format(color[:3]))
            self.pc.addWidget(self.PCASortBtns[i],row=6+i%5,col=5-int(i/5))
        

        # PCA used
        self.PCALabel = QtGui.QLabel()
        self.PCALabel.setText("PCA:")
        self.PCALabel.setFixedWidth(70)
        self.pc.addWidget(self.PCALabel,row=12,col=0,colspan=3) 

        self.PCAusedList = QtGui.QComboBox()
        self.PCAusedList.setFixedWidth(70)
        self.PCAusedList.addItems(self.PCAused)
        self.PCAusedList.currentIndexChanged.connect(self.__useThisPCA)
        self.pc.addWidget(self.PCAusedList,row=12,col=3,colspan=3)

        # load button
        loadBtn = QtGui.QPushButton("Load")
        # loadBtn.setMaximumWidth(70)  
        loadBtn.setFixedWidth(65)      
        loadBtn.clicked.connect(self.__load)
        self.pc.addWidget(loadBtn,row=13,col=0,colspan=3)

        # reset button
        resetBtn = QtGui.QPushButton("ResetAll")
        # loadBtn.setMaximumWidth(70)  
        resetBtn.setFixedWidth(80)      
        resetBtn.clicked.connect(self.__reset)
        self.pc.addWidget(resetBtn,row=13,col=2,colspan=2)

        # save button
        saveBtn = QtGui.QPushButton("Save")
        # loadBtn.setMaximumWidth(70)  
        saveBtn.setFixedWidth(65)      
        saveBtn.clicked.connect(self.__save)
        self.pc.addWidget(saveBtn,row=13,col=4,colspan=2)
        self.win.show()
        sys.exit(app.exec_())
        
    def __reset(self):
        if hasattr(self,"selectChan"):
            self.autoSortThisCheck.setChecked(False)
            self.saveChannelCheck.setChecked(False)
            self.chnResultPool.clear()
            self.chnList.clear()
            self.chnList.addItems(self.chns)

    def __load(self):
        if hasattr(self,"chnResultPool") and (len(self.chnResultPool)>0):
            print("do you want to discard result")
        else:
            # Load a file-selection dialog
            name = QtGui.QFileDialog.getOpenFileName(filter="h5 (*.h5)")
            file_name = name[0]
            if len(file_name) > 0:
                self.file_name = file_name
                self.window_info = "spike sorting: {0}".format(os.path.basename(self.file_name))
                self.win.setWindowTitle(self.window_info)
                if hasattr(self,'selectChan'):
                    delattr(self,'selectChan')
                self.chnList.clear()
                self.chns = []
                with hp.File(self.file_name,'r') as f:
                    for key in f.keys():
                        if key=="spikes":
                            for chn_unit in f["spikes"].keys():
                                self.chns.append(chn_unit.split('_')[1])
                if self.chns:
                    # self.autoSortAllCheck.setChecked(False)
                    self.autoSortThisCheck.setChecked(False)
                    self.saveChannelCheck.setChecked(False)
                    # init isAutoSortPool
                    self.chnResultPool = dict()                
                    self.chns = list(set(self.chns))
                    self.chns = [int(ite) for ite in self.chns]
                    self.chns.sort()
                    self.chns = [str(ite) for ite in self.chns]
                    self.chnList.addItems(self.chns)
                    # self.chnList.setCurrentIndex(0)
                    # self.__selectChan()
                    # add LineSegmentROI to pk0
                    if hasattr(self,"pk0_roi0"):
                        self.pk0.removeItem(self.pk0_roi0)
                        self.pk0.removeItem(self.pk0_roi1)
                        delattr(self,"pk0_roi0")
                        delattr(self,"pk0_roi1")
                    
                    self.pk0_roi0 = pg.LineSegmentROI([self.pk0_roi0_pos0, self.pk0_roi0_pos1],[0,0],pen='r')
                    self.pk0.addItem(self.pk0_roi0)
                    self.pk0_roi0.setZValue(10)
                    self.pk0_roi1 = pg.LineSegmentROI([self.pk0_roi1_pos0, self.pk0_roi1_pos1],[0,0],pen='r')
                    self.pk0.addItem(self.pk0_roi1)
                    self.pk0_roi1.setZValue(10)
                    
                    # self.__selectChan()
                    # add PolyLineROI to pk2
                    if hasattr(self,"pk2_roi"):
                        self.pk2.removeItem(self.pk2_roi)
                        delattr(self,"pk2_roi")
                    self.waveforms_pca_max = self.wavePCAs[:,0].max()
                    pk2_roi_pos_init = [[0,0],[0,self.waveforms_pca_max/3],[self.waveforms_pca_max/3,self.waveforms_pca_max/3],[self.waveforms_pca_max/3,0]]
                    self.pk2_roi = pg.PolyLineROI(pk2_roi_pos_init, pen='r',closed=True)
                    self.pk2_roi.setZValue(10)
                    self.pk2.addItem(self.pk2_roi)
                else:
                    print("There is no spike channels in this file.")

    def __autoShapePress(self,unit_id):
        def press():
            if hasattr(self,"selectChan"):
                self.saveChannelCheck.setChecked(False)
                self.chnResultPool.pop(self.selectChan,None)
                self.shapeSortBtns[unit_id].setStyleSheet("background-color: rgb{0};border-radius: 6px;".format((0,0,0)))
                pos = self.pk0_roi0.getLocalHandlePositions()[0][1]+self.pk0_roi0.pos()
                pk0_roi0_h0 = [pos[0],pos[1]]
                pos = self.pk0_roi0.getLocalHandlePositions()[1][1]+self.pk0_roi0.pos()
                pk0_roi0_h1 = [pos[0],pos[1]]

                pos = self.pk0_roi1.getLocalHandlePositions()[0][1]+self.pk0_roi1.pos()
                pk0_roi1_h0 = [pos[0],pos[1]]
                pos = self.pk0_roi1.getLocalHandlePositions()[1][1]+self.pk0_roi1.pos()
                pk0_roi1_h1 = [pos[0],pos[1]]
                
                selected_indexs_pk0 = self.__indexs_select_pk0(pk0_roi0_h0,pk0_roi0_h1,pk0_roi1_h0,pk0_roi1_h1)
                self.units[selected_indexs_pk0] = unit_id
                # self.units_pk0 = self.units[self.indexs_pk0]

                self.__draw_pk3()
                self.__update_pk3_roi()
                self.__draw_pk2()     
                if self.pca_3d is True:
                    self.__draw_3D_PCA()       

        return press
    def __indexs_select_pk0(self,pk0_roi0_h0,pk0_roi0_h1,pk0_roi1_h0,pk0_roi1_h1):
        # get indexs of selected waveforms in pk0
        spk_in_line = np.apply_along_axis(self.__in_select_line,1,self.waveforms_pk0,pk0_roi0_h0,pk0_roi0_h1)
        changed_index = np.where(spk_in_line==True)[0]
        changed_index = np.array(changed_index,dtype=np.int32)

        spk_in_line1 = np.apply_along_axis(self.__in_select_line,1,self.waveforms_pk0,pk0_roi1_h0,pk0_roi1_h1)
        changed_index1 = np.where(spk_in_line1==True)[0]
        changed_index1 = np.array(changed_index1,dtype=np.int32)
        changed_index =  np.intersect1d(changed_index, changed_index1)
        return changed_index + self.indexs_pk0[0]
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

    def __autoShapeRelease(self,unit_id):
        def release():
            if hasattr(self,"selectChan"):
                time.sleep(0.15)
                self.shapeSortBtns[unit_id].setStyleSheet("background-color: rgb{0};border-radius: 6px;".format(self.colors[unit_id][:3]))
        return release        

    def __autoPCAPress(self,unit_id):
        def press():
            if hasattr(self,"selectChan"):
                self.saveChannelCheck.setChecked(False)
                self.chnResultPool.pop(self.selectChan,None)

                self.PCASortBtns[unit_id].setStyleSheet("background-color: rgb{0};border-radius: 6px;".format((0,0,0)))
                pk2_roi_pos = []
                for hand in self.pk2_roi.getHandles():
                    tem_pos = hand.pos()+self.pk2_roi.pos()
                    pk2_roi_pos.append([tem_pos[0],tem_pos[1]])
                pk2_roi_pos = np.array(pk2_roi_pos)

                selected_indexs_pk2 = self.__indexs_select_pk2(pk2_roi_pos)
                self.units[selected_indexs_pk2] = unit_id
                # self.units_pk0 = self.units[self.indexs_pk0]

                self.__draw_pk3()
                self.__update_pk3_roi()
                self.__draw_pk2()      
                if self.pca_3d is True:
                    self.__draw_3D_PCA()      
        return press
    def __indexs_select_pk2(self,pk2_roi_pos):
        x_min = pk2_roi_pos[:,0].min()
        x_max = pk2_roi_pos[:,0].max()
        y_min = pk2_roi_pos[:,1].min()
        y_max = pk2_roi_pos[:,1].max()
        pca_1,pca_2 = self.PCAusedList.currentText().split("-")
        pca_1 = np.int(pca_1)-1
        pca_2 = np.int(pca_2)-1
        x = np.logical_and(self.wavePCAs[:,pca_1]>x_min, \
                            self.wavePCAs[:,pca_1]<x_max)
        y = np.logical_and(self.wavePCAs[:,pca_2]>y_min, \
                            self.wavePCAs[:,pca_2]<y_max)
        ind_0 = np.logical_and(x, y)
        ind_0 = np.where(ind_0 == True)[0]
        ind_0 = np.array(ind_0,dtype=np.int32)
        if ind_0.shape[0]>0:
            segments = []
            for i in range(pk2_roi_pos.shape[0]-1):
                segments.append([pk2_roi_pos[i],pk2_roi_pos[i+1]])
            segments.append([pk2_roi_pos[-1],pk2_roi_pos[0]])
            segments = np.array(segments)
            temp_pcas = self.wavePCAs[ind_0]
            temp_pcas = temp_pcas[:,[pca_1,pca_2]]
            is_intersect = np.apply_along_axis(self.__intersect_roi2,1,temp_pcas,segments,pca_1)
            return ind_0[is_intersect]
        else:
            return np.array([],dtype=np.int32)

    def __intersect_roi2(self,temp_point,segments,pca_1):
        point_B_xs_min = segments[:,:,0].min(axis=1)
        point_B_xs_max = segments[:,:,0].max(axis=1)
        point_B_ys_min = segments[:,:,1].min(axis=1)
        point_B_ys_max = segments[:,:,1].max(axis=1)
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
            if temp_point[pca_1]<seg[:,0].min():
                tem_insect.append(True)
            elif self.__ccw(A,C,D) != self.__ccw(B,C,D) and self.__ccw(A,B,C) != self.__ccw(A,B,D):
                tem_insect.append(True)
        if sum(tem_insect)%2==1:
            return True
        else:
            return False

    def __autoPCARelease(self,unit_id):
        def release():
            if hasattr(self,"selectChan"):
                time.sleep(0.15)
                self.PCASortBtns[unit_id].setStyleSheet("background-color: rgb{0};border-radius: 6px;".format(self.colors[unit_id][:3]))
        return release        
    
    # def __update_pk0_roi0(self):
    #     print "pk0 roi0"
    # def __update_pk0_roi1(self):
    #     print "pk0 roi1"

    def __save(self):
        if hasattr(self,"chns"):
            chnNotProcessed = set(self.chns).difference(set(self.chnResultPool.keys()))
            chnNotProcessed = sorted([int(ite) for ite in chnNotProcessed])
            self.unProcessedChns = copy.copy(chnNotProcessed)
            self.unProcessedChns = [str(ite) for ite in self.unProcessedChns]

            if len(chnNotProcessed)>0:
                err_msg = QtWidgets.QMessageBox()
                err_msg.setIcon(QtWidgets.QMessageBox.Information)
                err_msg.setText("Data in these channels are not processed: \n{0}. \n \n[save + autoSortUnprocessedChns]:  Results of processed channels will be saved, and unprocessed channels will be automatically sorted using the selected algorithm with parameters set in current control panel.\n\n[onlySaveProcessedChns]:  only save processed channel, and data in unprocessed channels remains unchanged.".format(chnNotProcessed))

                # autoBtn = QtGui.QPushButton()
                # autoBtn.setText("autoSort unprocessed chns")
                err_msg.addButton("save + autoSortUnprocessedChns",QtGui.QMessageBox.YesRole)
                err_msg.addButton("onlySaveProcessedChns",QtGui.QMessageBox.YesRole)
                err_msg.addButton("cancel",QtGui.QMessageBox.NoRole)                
                err_msg.buttonClicked.connect(self.__checkSave)
                # err_msg.setStandardButtons(QtWidgets.QMessageBox.Cancel)
                err_msg.exec_()
            else:
                err_msg = QtWidgets.QMessageBox()
                err_msg.setIcon(QtWidgets.QMessageBox.Information)
                err_msg.setText("Are you sure to save ?")                
                # autoBtn = QtGui.QPushButton()
                # autoBtn.setText("autoSort unprocessed chns")
                err_msg.addButton("Yes",QtGui.QMessageBox.YesRole)
                err_msg.addButton("cancel",QtGui.QMessageBox.NoRole)
                err_msg.buttonClicked.connect(self.__checkSave)
                # err_msg.setStandardButtons(QtWidgets.QMessageBox.Cancel)
                err_msg.exec_()
    
    def __checkSave(self,msg):
        if msg.text()=="onlySaveProcessedChns" or msg.text()=="Yes":
            with hp.File(self.file_name,"a") as f:
                for selectChan in self.chnResultPool.keys():
                    times,waveforms = self.__loadChnTimeWave(f,selectChan)
                    units = self.chnResultPool[selectChan]["units"]
                    if times is not None:
                        # delete current channel data in file
                        self.__deleteChn(f,selectChan)
                        # save sorted channel data in file
                        unit_unique = np.unique(units)
                        for ite_unit in unit_unique:
                            mask = units==ite_unit
                            temp_time = times[mask]
                            temp_waveform = waveforms[mask]
                            store_key_times = "spikes/spike_{0}_{1}/times".format(selectChan,ite_unit)
                            store_key_waveforms = "spikes/spike_{0}_{1}/waveforms".format(selectChan,ite_unit)
                            f[store_key_times] = temp_time
                            f[store_key_waveforms] = temp_waveform
                    f.flush()
            self.chnResultPool.clear()
            self.saveChannelCheck.setChecked(False)
            self.autoSortThisCheck.setChecked(False)

            self.__finish_pop()
        elif msg.text()=="save + autoSortUnprocessedChns":
            # save processed chn first
            with hp.File(self.file_name,"a") as f:
                for selectChan in self.chnResultPool.keys():
                    times,waveforms = self.__loadChnTimeWave(f,selectChan)
                    units = self.chnResultPool[selectChan]["units"]
                    if times is not None:
                        # delete current channel data in file
                        self.__deleteChn(f,selectChan)
                        # save sorted channel data in file
                        unit_unique = np.unique(units)
                        for ite_unit in unit_unique:
                            mask = units==ite_unit
                            temp_time = times[mask]
                            temp_waveform = waveforms[mask]
                            store_key_times = "spikes/spike_{0}_{1}/times".format(selectChan,ite_unit)
                            store_key_waveforms = "spikes/spike_{0}_{1}/waveforms".format(selectChan,ite_unit)
                            f[store_key_times] = temp_time
                            f[store_key_waveforms] = temp_waveform
                    f.flush()
            # process and save unprocessed chn
            # mintemp = float(self.mintempWgt.value())
            # maxtemp = float(self.maxtempWgt.value())
            # tempstep = self.SPCtempstep
            # dimensions = int(self.DimensionsWgt.value())
            # SWCycles = int(self.SWCyclesWgt.value())
            # KNeighbours = int(self.KNeighboursWgt.value())
            # UnitsNum = int(self.unitsNumWgt.currentText())
            # min_clus = 60

            with hp.File(self.file_name,"a") as f:
                for selectChan in self.unProcessedChns:
                    times,waveforms = self.__loadChnTimeWave(f,selectChan)
                    if times is not None:
                        waveletspk = self.__SubDoWavelets(waveforms)
                        exe_path = os.path.dirname(os.path.realpath(__file__))
                        tempFilePath = os.getcwd()
                        onlyfiles = [ite_file for ite_file in os.listdir(tempFilePath) if os.path.isfile(os.path.join(tempFilePath,ite_file))]
                        self.__deleteChn(f,selectChan)
                        # system_type = platform.system()
                        # if system_type == "Darwin":
                        #     exe_srcfile = os.path.join(exe_path,"SPC_{0}.exe".format(system_type))
                        #     exe_dstfile = os.path.join(tempFilePath,"SPC_{0}.exe".format(system_type))
                        # if "SPC_{0}.exe".format(system_type) not in onlyfiles:
                        #     shutil.copy(exe_srcfile,exe_dstfile)

                        system_type = platform.system()
                        system_architecture = platform.architecture()[0]
                        exe_srcfile = "err_neo"
                        exe_dstfile = "err_neo"
                        if system_type == "Darwin":
                            if system_architecture == "64bit":
                                exe_srcfile = os.path.join(exe_path,"SPC_{0}_{1}".format(system_type,system_architecture))
                                exe_dstfile = os.path.join(tempFilePath,"SPC_{0}_{1}".format(system_type,system_architecture))
                                if "SPC_{0}_{1}".format(system_type,system_architecture) not in onlyfiles:
                                    shutil.copy(exe_srcfile,exe_dstfile)
                        elif system_type == "Windows":
                            exe_srcfile = os.path.join(exe_path,"SPC_{0}_{1}.exe".format(system_type,system_architecture))
                            exe_dstfile = os.path.join(tempFilePath,"SPC_{0}_{1}.exe".format(system_type,system_architecture))
                            if "SPC_{0}_{1}.exe".format(system_type,system_architecture) not in onlyfiles:
                                shutil.copy(exe_srcfile,exe_dstfile)
                        elif system_type == "Linux":
                            exe_srcfile = os.path.join(exe_path,"SPC_{0}_{1}".format(system_type,system_architecture))
                            exe_dstfile = os.path.join(tempFilePath,"SPC_{0}_{1}".format(system_type,system_architecture))
                            if "SPC_{0}_{1}".format(system_type,system_architecture) not in onlyfiles:
                                shutil.copy(exe_srcfile,exe_dstfile)


                        # if system_type == "Darwin":
                        #     if system_architecture == "64bit":
                        #         exe_srcfile = os.path.join(exe_path,"SPC_{0}_{1}".format(system_type,system_architecture))
                        #         exe_dstfile = os.path.join(tempFilePath,"SPC_{0}_{1}".format(system_type,system_architecture))
                        #     if "SPC_{0}_{1}".format(system_type,system_architecture) not in onlyfiles:
                        #         shutil.copy(exe_srcfile,exe_dstfile)
                            
                        SpcConfigPath = os.path.join(tempFilePath,"SPC_config.run")
                        SpcTemDataPath = os.path.join(tempFilePath,"SpcTemData")
                        SpcOutDataPath = os.path.join(tempFilePath,"SpcOutData")

                        # mintemp = float(self.mintempWgt.value())
                        # maxtemp = float(self.maxtempWgt.value())
                        # tempstep = self.SPCtempstep
                        # dimensions = int(self.DimensionsWgt.value())
                        # SWCycles = int(self.SWCyclesWgt.value())
                        # KNeighbours = int(self.KNeighboursWgt.value())
                        # min_clus = 60

                        mintemp = float(self.mintempWgtt.text())
                        maxtemp = float(self.maxtempWgtt.text())
                        tempstep = self.SPCtempstep
                        SWCycles = int(self.SWCyclesWgtt.text())
                        KNeighbours = int(self.KNeighboursWgtt.text())
                        UnitsNum = int(self.unitsNumWgt.currentText())
                        min_clus = 60
                        dimensions = self.Dimensions

                        if os.path.isfile(SpcConfigPath):
                            os.remove(SpcConfigPath)
                        if os.path.isfile(SpcTemDataPath):
                            os.remove(SpcTemDataPath)
                        if os.path.isfile(SpcOutDataPath+".dg_01.lab"):
                            os.remove(SpcOutDataPath+".dg_01.lab")
                        if os.path.isfile(SpcOutDataPath+".dg_01"):        
                            os.remove(SpcOutDataPath+".dg_01")
                        if os.path.isfile(SpcOutDataPath+".mag"):        
                            os.remove(SpcOutDataPath+".mag")
                        if os.path.isfile(SpcOutDataPath+".mst{0}.edges".format(KNeighbours)):        
                            os.remove(SpcOutDataPath+".mst{0}.edges".format(KNeighbours))
                        if os.path.isfile(SpcOutDataPath+".param"):
                            os.remove(SpcOutDataPath+".param")

                        shuffle_idx = np.arange(waveletspk.shape[0])
                        np.random.shuffle(shuffle_idx)
                        waveletspk = waveletspk[shuffle_idx]

                        # save waveletspk into text file

                        np.savetxt(SpcTemDataPath,waveletspk)
                        # create SPC_config.run file
                        configList = [
                                        "NumberOfPoints: {0}".format(waveletspk.shape[0]),
                                        "DataFile: {0}".format("SpcTemData"),
                                        "OutFile: {0}".format("SpcOutData"),
                                        "Dimensions: {0}".format(dimensions),
                                        "MinTemp: {0}".format(mintemp),
                                        "MaxTemp: {0}".format(maxtemp),
                                        "TempStep: {0}".format(tempstep),
                                        "SWCycles: {0}".format(SWCycles),
                                        "KNearestNeighbours: {0}".format(KNeighbours),
                                        "MSTree|",
                                        "DirectedGrowth|",
                                        "SaveSuscept|",
                                        "WriteLables|",
                                        "WriteCorFile~"
                                        ]
                        configFile = open(SpcConfigPath,"w")
                        for ite in configList:
                            write_str = ite+"\n"
                            configFile.write(write_str)
                        configFile.close()

                        exe_str = "{0} {1}".format(exe_dstfile,os.path.basename(SpcConfigPath))
                        # # os.system(exe_str)
                        # if system_type=="Darwin":
                        #     print(os.system(exe_str))       
                        # elif system_type=="Windows":
                        #     subprocess.check_call(exe_str,shell=False)   
                        # elif system_type=="Linux":
                        #     print(os.system(exe_str))

                        # time.sleep(0.5)

                        is_spc = 0
                        if system_type=="Darwin":
                            is_spc = os.system(exe_str)       
                        elif system_type=="Windows":
                            is_spc = subprocess.call(exe_str,shell=False)   
                        elif system_type=="Linux":
                            is_spc = os.system(exe_str) 
                        time.sleep(0.5)

                        if is_spc == 0:

                            clu = np.loadtxt(SpcOutDataPath+".dg_01.lab")
                            tree = np.loadtxt(SpcOutDataPath+".dg_01")

                            temperature = self.__find_temperature(tree,mintemp,maxtemp,tempstep,min_clus)

                            auto_result = clu[temperature,2:]+1
                            auto_result = np.int32(auto_result)

                            sorted_idx = np.argsort(shuffle_idx)
                            auto_result = auto_result[sorted_idx]

                            # self.auto_result[self.auto_result>int(self.unitsNum[-1])] = 0
                            units = auto_result.copy()
                            units[units>int(self.unitsNumWgt.currentText())] = 0
                            
                            if os.path.isfile(SpcConfigPath):
                                os.remove(SpcConfigPath)
                                os.remove(SpcTemDataPath)
                                os.remove(SpcOutDataPath+".dg_01.lab")
                                os.remove(SpcOutDataPath+".dg_01")
                                os.remove(SpcOutDataPath+".mag")
                                os.remove(SpcOutDataPath+".mst{0}.edges".format(KNeighbours))
                                os.remove(SpcOutDataPath+".param")

                            # delete current channel data in file
                            self.__deleteChn(f,selectChan)
                            # save sorted channel data in file
                            unit_unique = np.unique(units)
                            for ite_unit in unit_unique:
                                mask = units==ite_unit
                                temp_time = times[mask]
                                temp_waveform = waveforms[mask]
                                store_key_times = "spikes/spike_{0}_{1}/times".format(selectChan,ite_unit)
                                store_key_waveforms = "spikes/spike_{0}_{1}/waveforms".format(selectChan,ite_unit)
                                f[store_key_times] = temp_time
                                f[store_key_waveforms] = temp_waveform
                            f.flush()

            self.chnResultPool.clear()
            self.saveChannelCheck.setChecked(False)
            self.autoSortThisCheck.setChecked(False)
            self.__finish_pop()
        elif msg.text()=="cancel":
            print("cancel")

    def __finish_pop(self):
        finish_msg = QtWidgets.QMessageBox()
        finish_msg.setIcon(QtWidgets.QMessageBox.Information)
        finish_msg.setText("Save Finished !")
        
        finish_msg.addButton("OK",QtGui.QMessageBox.YesRole)
        # err_msg.addButton("onlySaveProcessedChns",QtGui.QMessageBox.YesRole)
        finish_msg.exec_()

        # print(self.selectChan)

        self.__selectChan()

    def __deleteChn(self,f,selectChan):
        spk_startswith = "spike_{0}".format(selectChan)
        for chn in f['spikes'].keys():
            if chn.startswith(spk_startswith):
                del f['spikes'][chn]

    def __loadChnTimeWave(self,f,selectChan):
        times = list()
        waveforms = list()
        spk_startswith = "spike_{0}".format(selectChan)
        for chn_unit in f["spikes"].keys():
            if chn_unit.startswith(spk_startswith):
                time = f["spikes"][chn_unit]["times"].value
                waveform = f["spikes"][chn_unit]["waveforms"].value
                times.append(time)
                waveforms.append(waveform)
        if times:
            times = np.hstack(times)
            waveforms = np.vstack(waveforms)
            sort_index = np.argsort(times)
            waveforms = waveforms[sort_index]
            times = times[sort_index]
            return times,waveforms
        else:
            return None,None

    def __saveThisChannelCheck(self):
        if hasattr(self,"selectChan"):
            if self.saveChannelCheck.isChecked():
                self.chnResultPool[self.selectChan] = {"units":self.units,"isAutoSort":self.autoSortThisCheck.isChecked()}
                print(self.chnResultPool.keys())
            else:
                self.chnResultPool.pop(self.selectChan,None)

    def __autoSortThisCheck(self):
        if hasattr(self,"selectChan"):
            self.saveChannelCheck.setChecked(False)
            self.chnResultPool.pop(self.selectChan,None)
            if self.autoSortThisCheck.isChecked():
                if self.algorithmList.currentText()=="Wavelets&SPC":
                    self.__DoWaveletsSpc()
            else:
                self.units = self.units_backup
                self.__draw_pk3()
                self.__update_pk3_roi()
                self.__draw_pk2()
                if self.pca_3d is True:
                    self.__draw_3D_PCA()

            # else:
            #     if self.autoSortAllCheck.isChecked():
            #         self.autoSortAllCheck.setChecked(False)                 

    # def __autoSortAllCheck(self):
    #     if hasattr(self,"selectChan"):
    #         if not self.autoSortThisCheck.isChecked():
    #             self.autoSortThisCheck.setChecked(True)
            
    def __DoWaveletsSpc(self):
        if hasattr(self,"selectChan"):
            waveforms = self.__load_waveforms(self.selectChan,self.file_name)
            if waveforms is not None:
                waveletspk = self.__SubDoWavelets(waveforms)
                exe_path = os.path.dirname(os.path.realpath(__file__))
                tempFilePath = os.getcwd()
                onlyfiles = [ite_file for ite_file in os.listdir(tempFilePath) if os.path.isfile(os.path.join(tempFilePath,ite_file))]

                system_type = platform.system()
                system_architecture = platform.architecture()[0]
                exe_srcfile = "err_neo"
                exe_dstfile = "err_neo"
                if system_type == "Darwin":
                    if system_architecture == "64bit":
                        exe_srcfile = os.path.join(exe_path,"SPC_{0}_{1}".format(system_type,system_architecture))
                        exe_dstfile = os.path.join(tempFilePath,"SPC_{0}_{1}".format(system_type,system_architecture))
                        if "SPC_{0}_{1}".format(system_type,system_architecture) not in onlyfiles:
                            shutil.copy(exe_srcfile,exe_dstfile)
                elif system_type == "Windows":
                    exe_srcfile = os.path.join(exe_path,"SPC_{0}_{1}.exe".format(system_type,system_architecture))
                    exe_dstfile = os.path.join(tempFilePath,"SPC_{0}_{1}.exe".format(system_type,system_architecture))
                    if "SPC_{0}_{1}.exe".format(system_type,system_architecture) not in onlyfiles:
                        shutil.copy(exe_srcfile,exe_dstfile)
                elif system_type == "Linux":
                    exe_srcfile = os.path.join(exe_path,"SPC_{0}_{1}".format(system_type,system_architecture))
                    exe_dstfile = os.path.join(tempFilePath,"SPC_{0}_{1}".format(system_type,system_architecture))
                    if "SPC_{0}_{1}".format(system_type,system_architecture) not in onlyfiles:
                        shutil.copy(exe_srcfile,exe_dstfile)

                # if system_type == "Darwin":
                #     exe_srcfile = os.path.join(exe_path,"SPC_{0}.exe".format(system_type))
                #     exe_dstfile = os.path.join(tempFilePath,"SPC_{0}.exe".format(system_type))
                # if "SPC_{0}.exe".format(system_type) not in onlyfiles:
                #     shutil.copy(exe_srcfile,exe_dstfile)

                SpcConfigPath = os.path.join(tempFilePath,"SPC_config.run")
                SpcTemDataPath = os.path.join(tempFilePath,"SpcTemData")
                SpcOutDataPath = os.path.join(tempFilePath,"SpcOutData")

                # mintemp = float(self.mintempWgt.value())
                # maxtemp = float(self.maxtempWgt.value())
                # tempstep = self.SPCtempstep
                # SWCycles = int(self.SWCyclesWgt.value())
                # KNeighbours = int(self.KNeighboursWgt.value())
                # dimensions = int(self.DimensionsWgt.value())

                mintemp = float(self.mintempWgtt.text())
                maxtemp = float(self.maxtempWgtt.text())
                tempstep = self.SPCtempstep
                SWCycles = int(self.SWCyclesWgtt.text())
                KNeighbours = int(self.KNeighboursWgtt.text())
                min_clus = 60
                dimensions = self.Dimensions


                if os.path.isfile(SpcConfigPath):
                    os.remove(SpcConfigPath)
                if os.path.isfile(SpcTemDataPath):
                    os.remove(SpcTemDataPath)
                if os.path.isfile(SpcOutDataPath+".dg_01.lab"):
                    os.remove(SpcOutDataPath+".dg_01.lab")
                if os.path.isfile(SpcOutDataPath+".dg_01"):        
                    os.remove(SpcOutDataPath+".dg_01")
                if os.path.isfile(SpcOutDataPath+".mag"):        
                    os.remove(SpcOutDataPath+".mag")
                if os.path.isfile(SpcOutDataPath+".mst{0}.edges".format(KNeighbours)):        
                    os.remove(SpcOutDataPath+".mst{0}.edges".format(KNeighbours))
                if os.path.isfile(SpcOutDataPath+".param"):
                    os.remove(SpcOutDataPath+".param")

                shuffle_idx = np.arange(waveletspk.shape[0])
                np.random.shuffle(shuffle_idx)
                waveletspk = waveletspk[shuffle_idx]

                # save waveletspk into text file

                np.savetxt(SpcTemDataPath,waveletspk)
                # create SPC_config.run file
                configList = [
                                "NumberOfPoints: {0}".format(waveletspk.shape[0]),
                                "DataFile: {0}".format("SpcTemData"),
                                "OutFile: {0}".format("SpcOutData"),
                                "Dimensions: {0}".format(dimensions),
                                "MinTemp: {0}".format(mintemp),
                                "MaxTemp: {0}".format(maxtemp),
                                "TempStep: {0}".format(tempstep),
                                "SWCycles: {0}".format(SWCycles),
                                "KNearestNeighbours: {0}".format(KNeighbours),
                                "MSTree|",
                                "DirectedGrowth|",
                                "SaveSuscept|",
                                "WriteLables|",
                                "WriteCorFile~"
                                ]
                configFile = open(SpcConfigPath,"w")
                for ite in configList:
                    write_str = ite+"\n"
                    configFile.write(write_str)
                configFile.close()

                exe_str = "{0} {1}".format(exe_dstfile,os.path.basename(SpcConfigPath))
                is_spc = 0
                if system_type=="Darwin":
                    is_spc = os.system(exe_str)       
                elif system_type=="Windows":
                    is_spc = subprocess.call(exe_str,shell=False)   
                elif system_type=="Linux":
                    is_spc = os.system(exe_str) 
                time.sleep(0.5)

                if is_spc == 0:

                    clu = np.loadtxt(SpcOutDataPath+".dg_01.lab")
                    tree = np.loadtxt(SpcOutDataPath+".dg_01")

                    temperature = self.__find_temperature(tree,mintemp,maxtemp,tempstep,min_clus)

                    self.auto_result = clu[temperature,2:]+1
                    self.auto_result = np.int32(self.auto_result)

                    sorted_idx = np.argsort(shuffle_idx)
                    self.auto_result = self.auto_result[sorted_idx]

                    # self.auto_result[self.auto_result>int(self.unitsNum[-1])] = 0
                    self.units = self.auto_result.copy()
                    self.units[self.units>int(self.unitsNumWgt.currentText())] = 0
                    
                    self.__draw_pk3()
                    self.__update_pk3_roi()
                    self.__draw_pk2()
                    if self.pca_3d is True:
                        self.__draw_3D_PCA()

                    if os.path.isfile(SpcConfigPath):
                        os.remove(SpcConfigPath)
                        os.remove(SpcTemDataPath)
                        os.remove(SpcOutDataPath+".dg_01.lab")
                        os.remove(SpcOutDataPath+".dg_01")
                        os.remove(SpcOutDataPath+".mag")
                        os.remove(SpcOutDataPath+".mst{0}.edges".format(KNeighbours))
                        os.remove(SpcOutDataPath+".param")
                else:
                    self.__err_msg()
    def __err_msg(self):
        err_msg = QtWidgets.QMessageBox()
        err_msg.setIcon(QtWidgets.QMessageBox.Information)
        err_msg.setText("An error happened when trying to execute SPC.\nSince we use an independent execute file to do SPC, which need to create several temporary files in your computer.\nThis error may be due to permission denied or some incompatibility reasons.\nThe temporary file path can be see using command os.getcwd() in the python file you are running.\nYou can also contact use through zhangbo_1008@163.com \nWe are happy to provide help.\nBy the way, you can still sort spikes manually")        
        err_msg.addButton("Cancel",QtGui.QMessageBox.YesRole)
        # err_msg.addButton("onlySaveProcessedChns",QtGui.QMessageBox.YesRole)
        err_msg.exec_()

        # print(self.selectChan)

    def __SubDoWavelets(self,waveforms):
        scales = 4
        dimensions = 10
        nspk,ls = waveforms.shape
        cc = pywt.wavedec(waveforms,"haar",mode="symmetric",level=scales,axis=-1)
        cc = np.hstack(cc)

        sd = list()
        for i in range(ls):
            test_data = cc[:,i]
            thr_dist = np.std(test_data,ddof=1)*3
            thr_dist_min = np.mean(test_data)-thr_dist
            thr_dist_max = np.mean(test_data)+thr_dist
            aux = test_data[(test_data>thr_dist_min)&(test_data<thr_dist_max)]
            if aux.size > 10:
                sd.append(self.__test_ks(aux))
            else:
                sd.append(0)
        ind = np.argsort(sd)
        ind = ind[::-1]
        coeff = ind[:dimensions]
        
        waveletspk = cc[:,coeff]
        return waveletspk

    def __test_ks(self,x):
        x = x[~np.isnan(x)]
        n = x.size
        x.sort()
        yCDF = np.arange(1,n+1)/float(n)
        notdup = np.hstack([np.diff(x,1),[1]])
        notdup = notdup>0
        x_expcdf = x[notdup]
        y_expcdf = np.hstack([[0],yCDF[notdup]])
        zScores = (x_expcdf-np.mean(x))/np.std(x,ddof=1);
        mu = 0
        sigma = 1
        theocdf = 0.5*erfc(-(zScores-mu)/(np.sqrt(2)*sigma))

        delta1 = y_expcdf[:-1]-theocdf
        delta2 = y_expcdf[1:]-theocdf
        deltacdf = np.abs(np.hstack([delta1,delta2]))
        KSmax = deltacdf.max()
        return KSmax
    def __find_temperature(self,tree,mintemp,maxtemp,tempstep,min_clus):
        num_temp = int(floor(float(maxtemp-mintemp)/tempstep))
        aux = np.diff(tree[:,4])
        aux1 = np.diff(tree[:,5])
        aux2 = np.diff(tree[:,6])
        aux3 = np.diff(tree[:,7])
        temp=0;
        for t in range(0,num_temp-1):
            if(aux[t] > min_clus or aux1[t] > min_clus or aux2[t] > min_clus or aux3[t] >min_clus):
                temp=t+1

        if (temp==0 and tree[temp][5]<min_clus):
            temp=1

        return temp

    def __load_waveforms(self,selectChan,file_name):
        spk_startswith = "spike_{0}".format(selectChan)
        with hp.File(file_name,"r") as f:
            times = list()
            waveforms = list()
            for chn_unit in f["spikes"].keys():
                if chn_unit.startswith(spk_startswith):
                    tep_time = f["spikes"][chn_unit]["times"].value
                    waveform = f["spikes"][chn_unit]["waveforms"].value
                    times.append(tep_time)
                    waveforms.append(waveform)
            if times:
                times = np.hstack(times)
                waveforms = np.vstack(waveforms)
                sort_index = np.argsort(times)
                waveforms = waveforms[sort_index]
                return waveforms
            else:
                return None

    def __selectChan(self):
        self.__cleanPk0()
        self.__cleanPk2()
        self.__cleanPk3()
        print(self.chnList.currentText())
        if self.chnList.currentText():
            # if self.autoSortAllCheck.isChecked():
            #     self.autoSortThisCheck.setChecked(True)
            # else:
            #     # self.autoSortThisCheck.setAutoExclusive(False)
            #     self.autoSortThisCheck.setChecked(False)
            #     # self.autoSortThisCheck.setAutoExclusive(True)
            self.selectChan = self.chnList.currentText()            
            if self.selectChan in self.chnResultPool.keys():
                self.saveChannelCheck.setChecked(True)
            else:
                self.saveChannelCheck.setChecked(False)
            
            if (self.selectChan in self.chnResultPool.keys()) and (self.chnResultPool[self.selectChan]["isAutoSort"] is True):
                self.autoSortThisCheck.setChecked(True)
            else:
                self.autoSortThisCheck.setChecked(False)

            print("channel {0} is selected.".format(self.selectChan))
            self.times,self.units_backup,self.waveforms_range,self.wavePCAs = self.__load_chn_data(self.selectChan,self.file_name)
            if self.selectChan in self.chnResultPool.keys():
                self.units = self.chnResultPool[self.selectChan]["units"]
            else:
                self.units = self.units_backup.copy()

            if self.times is not None:
                self.pca_dot_size = self.wavePCAs.max()/150.0
                if self.pca_3d is True:
                    self.win2.opts['distance'] = abs(self.wavePCAs.max())*1.5

                self.__draw_pk3()
                self.__addROI2Pk3()
                # draw waveform in pk0
                self.__update_pk3_roi()
                # draw pca in pk2
                self.__useThisPCA()
                if self.pca_3d is True:
                    self.__draw_3D_PCA()
            else:
                err_msg = QtWidgets.QMessageBox()
                err_msg.setIcon(QtWidgets.QMessageBox.Information)
                err_msg.setText("There is no data in channel {0}".format(self.selectChan))
                err_msg.setStandardButtons(QtWidgets.QMessageBox.Cancel)
                err_msg.exec_()

    def __addROI2Pk3(self):
        # add pg.LinearRegionItem to pk3
        if hasattr(self,"pk3_roi"):
            self.pk3.removeItem(self.pk3_roi)
            delattr(self,"pk3_roi")
        self.pk3_roi = pg.LinearRegionItem()
        self.pk3_roi.setRegion([self.times[0],self.times[0]+8000])
        self.pk3_roi.sigRegionChangeFinished.connect(self.__update_pk3_roi)
        self.pk3.addItem(self.pk3_roi,ignoreBounds=True)

    def __update_pk3_roi(self):
        self.__loadWaveInTimebin()
        if self.units is not None:
            self.__draw_pk0()
    def __draw_pk0(self):
        self.__cleanPk0()
        if self.indexs_pk0 is not None:
            units_pk0 = self.units[self.indexs_pk0]
            unique_units = np.unique(units_pk0)
            unique_units = unique_units.tolist()
            self.wavesLinesItem = []
            for i,ite_unit in enumerate(unique_units):
                mask = units_pk0==ite_unit
                tempWave = self.waveforms_pk0[mask]
                temp_time = np.ones(tempWave.shape,dtype=np.uint16)*np.arange(tempWave.shape[1],dtype=np.uint16)
                self.wavesLinesItem.append(MultiLine(temp_time,tempWave,self.colors[ite_unit]))
                self.pk0.addItem(self.wavesLinesItem[i])

    def __loadWaveInTimebin(self):
        minX,maxX = self.pk3_roi.getRegion()
        waveforms = list()
        times = list()
        masks = list()
        with hp.File(self.file_name,"r") as f:
            spk_startswith = "spike_{0}".format(self.selectChan)
            for chn_unit in f["spikes"].keys():
                if chn_unit.startswith(spk_startswith):
                    tep_time = f["spikes"][chn_unit]["times"].value
                    mask = (tep_time > minX) & (tep_time < maxX)
                    waveform = f["spikes"][chn_unit]["waveforms"].value[mask]
                    masks.append(mask)
                    times.append(tep_time)
                    waveforms.append(waveform)
            if times:
                times = np.hstack(times)
                masks = np.hstack(masks)
                waveforms = np.vstack(waveforms)
                sorted_temp = np.argsort(times)
                masks_temp = masks[sorted_temp]
                units = self.units[masks_temp]
                self.indexs_pk0 = np.arange(self.units.shape[0])
                self.indexs_pk0 = self.indexs_pk0[masks_temp]
                times = times[masks]
                sorted_temp = np.argsort(times)
                # times = times[sorted_temp]
                waveforms = waveforms[sorted_temp]
                self.waveforms_pk0 = waveforms
                # self.units_pk0 = units
                # self.units_pk0 = self.units[self.indexs_pk0]
                # return waveforms,units
            else:
                self.waveforms_pk0 = None
                # self.units_pk0 = None
                # return None, None

    def __load_chn_data(self,selectChan,file_name):
        spk_startswith = "spike_{0}".format(selectChan)
        with hp.File(file_name,"r") as f:
            times = list()
            waveforms = list()
            units = list()
            for chn_unit in f["spikes"].keys():
                if chn_unit.startswith(spk_startswith):
                    tep_time = f["spikes"][chn_unit]["times"].value
                    waveform = f["spikes"][chn_unit]["waveforms"].value
                    unit = int(chn_unit.split("_")[-1])
                    unit = np.ones(tep_time.shape,dtype=np.int32)*unit
                    times.append(tep_time)
                    waveforms.append(waveform)
                    units.append(unit)
            if times:
                times = np.hstack(times)
                units = np.hstack(units)
                waveforms = np.vstack(waveforms)
                sort_index = np.argsort(times)
                units = units[sort_index]
                waveforms = waveforms[sort_index]
                times = times[sort_index]
                # calculate waveform_range 
                waveforms_max = np.apply_along_axis(max,1,waveforms)
                waveforms_min = np.apply_along_axis(min,1,waveforms)
                waveforms_range = np.vstack([waveforms_min,waveforms_max]).T
                # calculate PCA of waveforms
                scaler = StandardScaler()
                scaler.fit(waveforms)
                waveforms_scaled = scaler.transform(waveforms)
                pca = PCA(n_components=self.pca_used_num)
                pca.fit(waveforms_scaled)
                wavePCAs = pca.transform(waveforms_scaled)
                return times,units,waveforms_range,wavePCAs
            else:
                return None,None,None,None

    def __useThisPCA(self):
        if hasattr(self,"selectChan"):
            self.__draw_pk2()

    def __draw_pk2(self):
        self.__cleanPk2()
        if self.units is not None:
            unique_units = np.unique(self.units)
            unique_units = unique_units.tolist()
            pca_1,pca_2 = self.PCAusedList.currentText().split("-")
            pca_1 = np.int(pca_1)-1
            pca_2 = np.int(pca_2)-1
            if self.wavePCAs[0].shape[0]>2:
                xs = self.wavePCAs[:,pca_1]
                ys = self.wavePCAs[:,pca_2]
                self.PcaScatterItem = []
                seg_num = 5000
                for i,ite_unit in enumerate(unique_units):
                    mask = self.units==ite_unit
                    temp_xs = xs[mask]
                    temp_ys = ys[mask]
                    segs = int(ceil(temp_xs.shape[0]/float(seg_num)))
                    for j in range(segs):
                        temp_xs_j = temp_xs[j*seg_num:(j+1)*seg_num]
                        temp_ys_j = temp_ys[j*seg_num:(j+1)*seg_num]
                        self.PcaScatterItem.append(pg.ScatterPlotItem(temp_xs_j,temp_ys_j,pen=self.colors[ite_unit],brush=self.colors[ite_unit],size=3,symbol="o"))
                for i in range(len(self.PcaScatterItem)):
                    self.pk2.addItem(self.PcaScatterItem[i])


    def __draw_pk3(self):
        self.__cleanPk3()
        unique_units = np.unique(self.units)
        unique_units = unique_units.tolist()
        self.timestampLinesItem = list()
        for i, ite_unit in enumerate(unique_units):
            mask = self.units==ite_unit
            temp_waveRange = self.waveforms_range[mask]
            temp_time = self.times[mask]
            temp_time = np.ones(temp_waveRange.shape,dtype=np.int32)*temp_time[:,np.newaxis]
            self.timestampLinesItem.append(MultiLine(temp_time,temp_waveRange,self.colors[ite_unit]))
            self.pk3.addItem(self.timestampLinesItem[i])

    def __draw_3D_PCA(self):
        self.__clean_3D_PCA()
        unique_units = np.unique(self.units)
        self.PCA3DLinesItem=list()        
        for i, unit in enumerate(unique_units):
            color = list(self.colors[i])
            color[0] = color[0]/255.0
            color[1] = color[1]/255.0
            color[2] = color[2]/255.0
            color[3] = 0.5
            self.PCA3DLinesItem.append(self.gl.GLScatterPlotItem(pos=self.wavePCAs[self.units==unit],size=self.pca_dot_size,color=color,pxMode=False))
            self.win2.addItem(self.PCA3DLinesItem[i])
    def __clean_3D_PCA(self):
        if hasattr(self,"PCA3DLinesItem"):
            for i in range(len(self.PCA3DLinesItem)):
                self.win2.removeItem(self.PCA3DLinesItem[i])
            delattr(self,"PCA3DLinesItem")
    def __cleanPk0(self):
        if hasattr(self,"wavesLinesItem"):
            for i in range(len(self.wavesLinesItem)):
                self.pk0.removeItem(self.wavesLinesItem[i])
            delattr(self,"wavesLinesItem")

    def __cleanPk2(self):
        if hasattr(self,"PcaScatterItem"):
            for i in range(len(self.PcaScatterItem)):
                self.pk2.removeItem(self.PcaScatterItem[i])
            delattr(self,"PcaScatterItem")

    def __cleanPk3(self):
        if hasattr(self,"timestampLinesItem"):
            for i in range(len(self.timestampLinesItem)):
                self.pk3.removeItem(self.timestampLinesItem[i])
            delattr(self,"timestampLinesItem")

    def __useThisAlgorithm(self):
        self.saveChannelCheck.setChecked(False)
        if hasattr(self,"selectChan"):
            self.chnResultPool.pop(self.selectChan,None)        
        self.autoSortThisCheck.setChecked(False)

        self.algorithmArgTree.clear()
        if self.algorithmList.currentText()=="Wavelets&SPC":
            self.__WaveletsSPCInit()
        elif self.algorithmList.currentText()=="test":
            print("test")
        
        if hasattr(self,"selectChan"):
            self.units = self.units_backup
            self.__draw_pk3()
            self.__update_pk3_roi()
            self.__draw_pk2()
            if self.pca_3d is True:
                self.__draw_3D_PCA()

    def __WaveletsSPCInit(self):
        self.mintemp = 0
        self.maxtemp = 0.201
        self.SPCtempstep = 0.01
        self.min_clus = 60
        self.KNeighbours = 11
        self.SWCycles = 100
        self.Dimensions = 10

        i7 = pg.TreeWidgetItem(["Default"])
        self.defaultSPC = QtGui.QPushButton()
        self.defaultSPC.clicked.connect(self.__defaultSPC)
        self.defaultSPC.setFixedWidth(60)
        self.defaultSPC.setText("Click")
        i7.setWidget(1,self.defaultSPC)
        self.algorithmArgTree.addTopLevelItem(i7)

        # i1  = pg.TreeWidgetItem(["MinTemp"])
        # self.mintempWgt = pg.SpinBox(value=self.mintemp,step=0.01)
        # self.mintempWgt.setMinimum(0)
        # self.mintempWgt.setFixedWidth(60)
        # self.mintempWgt.sigValueChanged.connect(self.__autoParamChanged)
        # i1.setWidget(1, self.mintempWgt)
        # self.algorithmArgTree.addTopLevelItem(i1)
        # i2  = pg.TreeWidgetItem(["MaxTemp"])
        # self.maxtempWgt = pg.SpinBox(value=self.maxtemp,step=0.01)
        # self.maxtempWgt.setMinimum(0.01)
        # self.maxtempWgt.setFixedWidth(60)
        # self.maxtempWgt.sigValueChanged.connect(self.__autoParamChanged)
        # i2.setWidget(1, self.maxtempWgt)
        # self.algorithmArgTree.addTopLevelItem(i2)
        # i3  = pg.TreeWidgetItem(["SWCycles"])
        # self.SWCyclesWgt = pg.SpinBox(value=self.SWCycles,step=1)
        # self.SWCyclesWgt.setMinimum(1)
        # self.SWCyclesWgt.setFixedWidth(60)
        # self.SWCyclesWgt.sigValueChanged.connect(self.__autoParamChanged)
        # i3.setWidget(1, self.SWCyclesWgt)
        # self.algorithmArgTree.addTopLevelItem(i3)
        # i4  = pg.TreeWidgetItem(["KNeighbours"])
        # self.KNeighboursWgt = pg.SpinBox(value=self.KNeighbours,step=1)
        # self.KNeighboursWgt.setMinimum(1)
        # self.KNeighboursWgt.setFixedWidth(60)
        # self.KNeighboursWgt.sigValueChanged.connect(self.__autoParamChanged)
        # i4.setWidget(1, self.KNeighboursWgt)
        # self.algorithmArgTree.addTopLevelItem(i4)
        # i5  = pg.TreeWidgetItem(["Dimensions"])
        # self.DimensionsWgt = pg.SpinBox(value=self.Dimensions,step=1)
        # self.DimensionsWgt.setMinimum(1)
        # self.DimensionsWgt.setFixedWidth(60)
        # self.DimensionsWgt.sigValueChanged.connect(self.__autoParamChanged)
        # i5.setWidget(1, self.DimensionsWgt)
        # self.algorithmArgTree.addTopLevelItem(i5)

        i11 = pg.TreeWidgetItem(["MinTemp"])
        self.mintempWgtt = QtGui.QPushButton()
        self.mintempWgtt.setText(str(self.mintemp))
        self.mintempWgtt.setFixedWidth(60)
        self.mintempWgtt.setStyleSheet("background-color: rgb(150,150,150);border-radius: 6px;")
        self.mintempWgtt.clicked.connect(self.__setMinTemp)
        i11.setWidget(1,self.mintempWgtt)
        self.algorithmArgTree.addTopLevelItem(i11)

        i22 = pg.TreeWidgetItem(["MaxTemp"])
        self.maxtempWgtt = QtGui.QPushButton()
        self.maxtempWgtt.setText(str(self.maxtemp))
        self.maxtempWgtt.setFixedWidth(60)
        self.maxtempWgtt.setStyleSheet("background-color: rgb(150,150,150);border-radius: 6px;")
        self.maxtempWgtt.clicked.connect(self.__setMaxTemp)
        i22.setWidget(1,self.maxtempWgtt)
        self.algorithmArgTree.addTopLevelItem(i22)

        i33 = pg.TreeWidgetItem(["Cycles"])
        self.SWCyclesWgtt = QtGui.QPushButton()
        self.SWCyclesWgtt.setText(str(self.SWCycles))
        self.SWCyclesWgtt.setFixedWidth(60)
        self.SWCyclesWgtt.setStyleSheet("background-color: rgb(150,150,150);border-radius: 6px;")
        self.SWCyclesWgtt.clicked.connect(self.__setSWCycles)
        i33.setWidget(1,self.SWCyclesWgtt)
        self.algorithmArgTree.addTopLevelItem(i33)

        i44 = pg.TreeWidgetItem(["KNeighbors"])
        self.KNeighboursWgtt = QtGui.QPushButton()
        self.KNeighboursWgtt.setText(str(self.KNeighbours))
        self.KNeighboursWgtt.setFixedWidth(60)
        self.KNeighboursWgtt.setStyleSheet("background-color: rgb(150,150,150);border-radius: 6px;")
        self.KNeighboursWgtt.clicked.connect(self.__setKNeighbours)
        i44.setWidget(1,self.KNeighboursWgtt)
        self.algorithmArgTree.addTopLevelItem(i44)

        i6  = pg.TreeWidgetItem(["UnitsNum"])
        self.unitsNumWgt = QtGui.QComboBox()
        self.unitsNumWgt.setFixedWidth(60)
        self.unitsNumWgt.addItems(self.unitsNum)
        self.unitsNumWgt.setCurrentIndex(3)
        self.unitsNumWgt.currentIndexChanged.connect(self.__unitsNumChanged)
        i6.setWidget(1, self.unitsNumWgt)
        self.algorithmArgTree.addTopLevelItem(i6)

        # i55 = pg.TreeWidgetItem(["Dimensions"])
        # self.DimensionsWgtt = QtGui.QPushButton()
        # self.DimensionsWgtt.setText(str(self.Dimensions))
        # self.DimensionsWgtt.setFixedWidth(60)
        # self.DimensionsWgtt.setStyleSheet("background-color: rgb(150,150,150);border-radius: 6px;")
        # self.DimensionsWgtt.clicked.connect(self.__setDimensions)
        # i55.setWidget(1,self.DimensionsWgtt)
        # self.algorithmArgTree.addTopLevelItem(i55)

        # self.defaultSPCisClick = False

    # parameters widget for SPC
    def __setMinTemp(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Set minimal temperature value.\n0 <= MinTemp < MaxTemp\nDefault is 0.')
        w.textValueSelected.connect(self.__setMinTemp_value)
        w.exec_()
    def __setMaxTemp(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Set maximum temperature value.\nMaxTemp >= MinTemp\nDefault is 0.201.')
        w.textValueSelected.connect(self.__setMaxTemp_value)
        w.exec_()
    def __setSWCycles(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Cycles of SPC\nThis value should be a positive integer\nDefault is 100.')
        w.textValueSelected.connect(self.__setSWCycles_value)
        w.exec_()
    def __setKNeighbours(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('K-nearest neighbors\nThis value should be a positive integer\nDefault is 11.')
        w.textValueSelected.connect(self.__setKNeighbours_value)
        w.exec_()
    # def __setDimensions(self):
    #     print int(self.DimensionsWgtt.text())

    def __setMinTemp_value(self,value):
        if ftl.is_num(value) and (float(value)>=0) and (float(value)!=float(self.mintempWgtt.text())):
            self.mintempWgtt.setText(value)
            self.__autoParamChanged()
    def __setMaxTemp_value(self,value):
        if ftl.is_num(value) and (float(value)>0) and (float(value)!=float(self.maxtempWgtt.text())):
            self.maxtempWgtt.setText(value)
            self.__autoParamChanged()
    def __setSWCycles_value(self,value):
        if ftl.is_num(value) and (int(value)>0) and (int(value)!=int(self.SWCyclesWgtt.text())):
            self.SWCyclesWgtt.setText(str(int(value)))
            self.__autoParamChanged()
    def __setKNeighbours_value(self,value):
        if ftl.is_num(value) and (int(value)>0) and (int(value)!=int(self.KNeighboursWgtt.text())):
            self.KNeighboursWgtt.setText(str(int(value)))
            self.__autoParamChanged()

    def __defaultSPC(self):
        # self.defaultSPCisClick = True
        # self.mintempWgt.setValue(self.mintemp)
        # self.maxtempWgt.setValue(self.maxtemp)
        # self.SWCyclesWgt.setValue(self.SWCycles)
        # self.KNeighboursWgt.setValue(self.KNeighbours)
        # self.DimensionsWgt.setValue(self.Dimensions)
        # self.unitsNumWgt.setCurrentIndex(3)
        self.mintempWgtt.setText(str(self.mintemp))
        self.maxtempWgtt.setText(str(self.maxtemp))
        self.SWCyclesWgtt.setText(str(self.SWCycles))
        self.KNeighboursWgtt.setText(str(self.KNeighbours))
        self.unitsNumWgt.setCurrentIndex(3)
        # time.sleep(0.3)
        # self.defaultSPCisClick = False

        if hasattr(self,"selectChan"):
            if self.autoSortThisCheck.isChecked():
                if self.algorithmList.currentText()=="Wavelets&SPC":
                    # if self.defaultSPCisClick is False:
                    self.__DoWaveletsSpc()
        
    def __autoParamChanged(self):
        if hasattr(self,"selectChan"):
            if self.autoSortThisCheck.isChecked():
                self.saveChannelCheck.setChecked(False)
                self.chnResultPool.pop(self.selectChan,None)
                if self.algorithmList.currentText()=="Wavelets&SPC":
                    # if self.defaultSPCisClick is False:
                    self.__DoWaveletsSpc()

    def __unitsNumChanged(self):
        if hasattr(self,"auto_result"):
            if self.autoSortThisCheck.isChecked():

                self.saveChannelCheck.setChecked(False)
                self.chnResultPool.pop(self.selectChan,None)

                self.units = self.auto_result.copy()
                self.units[self.units>int(self.unitsNumWgt.currentText())] = 0

                self.__draw_pk3()
                self.__update_pk3_roi()
                self.__draw_pk2()
                if self.pca_3d is True:
                    self.__draw_3D_PCA()

                
# The SpikeSorting class use this class to draw multiple lines quickly in a memory-efficient way.
class MultiLine(pg.QtGui.QGraphicsPathItem):
    def __init__(self, x, y,color):
        """x and y are 2D arrays of shape (Nplots, Nsamples)"""
        connect = np.ones(x.shape, dtype=bool)
        # don't draw the segment between each trace
        connect[:,-1] = 0 
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        pg.QtGui.QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen(color=color))
    # override because QGraphicsPathItem.shape is too expensive.
    def shape(self): 
        return pg.QtGui.QGraphicsItem.shape(self)
    def boundingRect(self):
        return self.path.boundingRect()
