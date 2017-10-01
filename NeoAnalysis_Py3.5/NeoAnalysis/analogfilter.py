# -*- coding: utf-8 -*-
'''
This is a module for filtering analog signal. The filter methods include band-pass and band-stop filtering.
This module can be used with or without a GUI window.
This work is based on:
    * Bo Zhang, Ji Dai - first version
'''
import copy
import imp
from . import pyqtgraph as pg
from .pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from .pyqtgraph import dockarea
import numpy as np
from scipy import signal
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
    def __init__(self,gui=True,reclaim_space=False,filename=None,channels=None,btype=None,ftype="butter",order=6,zerophase=True,**args):
        """
        Initialize the AnalogFilter class.

        if gui is False, these parameters need to be set to filtering data using script
            filename (string): path of file
            channels (str, list): channel name or channel list to be filtered. list ["23","24"]
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
                Default: 6
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

        """
        self.reclaim_space = reclaim_space
        # A gui window will be show if gui = True
        if gui is True:
            alpha=190
            self.colors=[(1*255,1*255,1*255,alpha),(0,1*255,0,alpha),(1*255,0,0,alpha),(1*255,1*255,0,alpha),(0,0,1*255,alpha),
                        (1*255,0,1*255,alpha),(0,1*255,1*255,alpha),(0,0.5*255,0.5*255,alpha),(0.5*255,0,0.5*255,alpha),(0.5*255,0.5*255,0,alpha)]
            self.color =[(1,1,1,0.5),            (0,1,0,0.5),    (1,0,0,0.5),    (1,1,0,0.5),       (0,0,1,0.5),
                        (1,0,1,0.5),        (0,1,1,0.5),        (0,0.5,0.5,0.5),        (0.5,0,0.5,0.5),        (0.5,0.5,0,0.5)]
            self.filterAlgorithms = ["bandpass","bandstop","lowpass","highpass"]
            self.filterAlgorithms.append("addNew")
            # create the main GUI window
            app = QtGui.QApplication([])
            win = QtGui.QMainWindow()
            win.setFixedWidth(1000)
            win.setFixedHeight(600) #700
            self.area = dockarea.DockArea()
            win.setCentralWidget(self.area)
            win.setWindowTitle("analog signal filtering")
            # create docks, place them into the window one at a time
            d0 = dockarea.Dock("part of signal",size=(750,450))  
            d1 = dockarea.Dock("control panel", size=(250,450))
            d2 = dockarea.Dock("signal", size=(1000,150))
            
            d0.setAcceptDrops(False)
            d1.setAcceptDrops(False)
            d2.setAcceptDrops(False)

            self.area.addDock(d0,"left")
            self.area.addDock(d2,"bottom",d0)
            self.area.addDock(d1,"right",d0)

            # add part signal drawing window to d0
            self.pk0 = pg.PlotWidget()
            # self.pk0.hideAxis('bottom')
            # self.pk0.hideAxis('left')
            self.pk0.setMouseEnabled(x=True,y=True)        
            d0.addWidget(self.pk0)

            # add signal drawing window to d2
            self.pk2 = pg.PlotWidget()
            self.pk2.setLabel("bottom","Time (ms)")
            # self.pk2.hideAxis('bottom')
            # self.pk2.hideAxis('left')
            self.pk2.setMouseEnabled(x=True,y=False)      
            d2.addWidget(self.pk2)

            # add control panel to d1
            self.pc = pg.LayoutWidget()
            d1.addWidget(self.pc)

            # channel list
            self.chnLabel = QtGui.QLabel()
            self.chnLabel.setText("Channel:")
            self.chnLabel.setFixedWidth(70)
            self.pc.addWidget(self.chnLabel,row=0,col=0,colspan=1) 
            
            self.chnList = QtGui.QComboBox()
            self.chnList.setFixedWidth(70)
            # self.chnList.setFixedHeight(30)
            self.chnList.currentIndexChanged.connect(self.__selectChan)
            self.pc.addWidget(self.chnList,row=0,col=1,colspan=2)

            # use filtering for this channel
            saveThisChannel = QtGui.QLabel()
            saveThisChannel.setText("SaveThisResult:")
            saveThisChannel.setFixedWidth(140)
            self.pc.addWidget(saveThisChannel,row=1,col=0,colspan=2)
            # self.saveChannelCheck = QtGui.QCheckBox()
            self.saveChannelCheck = QtGui.QCheckBox()
            self.saveChannelCheck.setFixedWidth(20)
            self.saveChannelCheck.clicked.connect(self.__saveThisChannelCheck)
            self.pc.addWidget(self.saveChannelCheck,row=1,col=2,colspan=1)

            # filtering this channel
            FilterThisChannel = QtGui.QLabel()
            FilterThisChannel.setText("FilterThisChn:")
            FilterThisChannel.setFixedWidth(140)
            self.pc.addWidget(FilterThisChannel,row=2,col=0,colspan=2)
            # self.saveChannelCheck = QtGui.QCheckBox()
            self.FilterChannelCheck = QtGui.QCheckBox()
            self.FilterChannelCheck.setFixedWidth(20)
            self.FilterChannelCheck.clicked.connect(self.__FilterThisChnCheck)
            self.pc.addWidget(self.FilterChannelCheck,row=2,col=2,colspan=1)

            # algorithms args
            self.algorithmArgTree = pg.TreeWidget()
            self.algorithmArgTree.setColumnCount(2)
            self.algorithmArgTree.setHeaderLabels(["param","value"])        
            self.pc.addWidget(self.algorithmArgTree,row=4,col=0,colspan=3)
            self.algorithmArgTree.clear()

            # algorithms list
            self.algorithmList = QtGui.QComboBox()
            self.algorithmList.setFixedWidth(140)
            self.algorithmList.currentIndexChanged.connect(self.__useThisAlgorithm)
            self.algorithmList.addItems(self.filterAlgorithms)
            self.pc.addWidget(self.algorithmList,row=3,col=0,colspan=3)

            # load button
            loadBtn = QtGui.QPushButton("Load")
            # loadBtn.setMaximumWidth(70)  
            loadBtn.setFixedWidth(65)      
            loadBtn.clicked.connect(self.__load)
            self.pc.addWidget(loadBtn,row=5,col=0,colspan=0)

            # reset button
            resetBtn = QtGui.QPushButton("ResetAll")
            # loadBtn.setMaximumWidth(70)  
            resetBtn.setFixedWidth(80)      
            resetBtn.clicked.connect(self.__reset)
            self.pc.addWidget(resetBtn,row=5,col=1,colspan=1)

            # save button
            saveBtn = QtGui.QPushButton("Save")
            # loadBtn.setMaximumWidth(70)  
            saveBtn.setFixedWidth(65)      
            saveBtn.clicked.connect(self.__save)
            self.pc.addWidget(saveBtn,row=5,col=2,colspan=1)
            win.show()
            sys.exit(app.exec_())
            win.show()
            sys.exit(app.exec_())
        elif gui is False:
            algorithm = btype
            chnInfo = {
                        "zerophase":zerophase,
                        "order":order,
                        "algorithm":algorithm,
                        "ftype":ftype}
            if algorithm == "bandpass":
                if ftype in ["butter","bessel"]:
                    lowcut = args["lowcut"]
                    highcut = args["highcut"]
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                elif ftype == "cheby1":
                    lowcut = args["lowcut"]
                    highcut = args["highcut"] 
                    rp = args["rp"]         
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                    chnInfo["rp"] = rp
                elif ftype == "cheby2":
                    lowcut = args["lowcut"]
                    highcut = args["highcut"]  
                    rs = args["rs"]     
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                    chnInfo["rs"] = rs
                elif ftype == "ellip":
                    lowcut = args["lowcut"]
                    highcut = args["highcut"]  
                    rp = args["rp"]
                    rs = args["rs"]
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                    chnInfo["rp"] = rp
                    chnInfo["rs"] = rs
            elif algorithm == "bandstop":
                if ftype in ["butter","bessel"]:
                    lowcut = args["lowcut"]
                    highcut = args["highcut"]
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                elif ftype == "cheby1":
                    lowcut = args["lowcut"]
                    highcut = args["highcut"]     
                    rp = args["rp"]      
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                    chnInfo["rp"] = rp
                elif ftype == "cheby2":
                    lowcut = args["lowcut"]
                    highcut = args["highcut"]  
                    rs = args["rs"]    
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                    chnInfo["rs"] = rs
                elif ftype == "ellip":
                    lowcut = args["lowcut"]
                    highcut = args["highcut"]  
                    rp = args["rp"]
                    rs = args["rs"]
                    chnInfo["lowcut"] = lowcut
                    chnInfo["highcut"] = highcut
                    chnInfo["rp"] = rp
                    chnInfo["rs"] = rs             
            elif algorithm == "highpass":
                if ftype in ["butter","bessel"]:
                    lowcut = args["lowcut"]
                    chnInfo["lowcut"] = lowcut
                elif ftype == "cheby1":
                    lowcut = args["lowcut"]
                    rp = args["rp"]      
                    chnInfo["lowcut"] = lowcut
                    chnInfo["rp"] = rp
                elif ftype == "cheby2":
                    lowcut = args["lowcut"]
                    rs = args["rs"]      
                    chnInfo["lowcut"] = lowcut
                    chnInfo["rs"] = rs
                elif ftype == "ellip":
                    lowcut = args["lowcut"]
                    rp = args["rp"]
                    rs = args["rs"]
                    chnInfo["lowcut"] = lowcut
                    chnInfo["rp"] = rp
                    chnInfo["rs"] = rs
            elif algorithm == "lowpass":
                if ftype in ["butter","bessel"]:
                    highcut = args["highcut"]
                    chnInfo["highcut"] = highcut
                elif ftype == "cheby1":
                    highcut = args["highcut"]
                    rp = args["rp"]     
                    chnInfo["highcut"] = highcut
                    chnInfo["rp"] = rp
                elif ftype == "cheby2":
                    highcut = args["highcut"]
                    rs = args["rs"]    
                    chnInfo["highcut"] = highcut
                    chnInfo["rs"] = rs
                elif ftype == "ellip":
                    highcut = args["highcut"]
                    rp = args["rp"]
                    rs = args["rs"]
                    chnInfo["highcut"] = highcut
                    chnInfo["rp"] = rp
                    chnInfo["rs"] = rs

            unProcessedChns = []
            if isinstance(channels,str):
                unProcessedChns = [channels]
            elif isinstance(channels,list):
                unProcessedChns = channels
            else:
                raise ValueError("Parameter channels should be list or str type, or it is not a valid channel")
            self.selectChan=True
            if unProcessedChns:
                with hp.File(filename,"a") as f:
                    for selectChan in unProcessedChns:
                        chn_key = "analog_{0}".format(selectChan)
                        sampling_rate = f["analogs"][chn_key]["sampling_rate"].value
                        chnInfo["fs"] = sampling_rate
                        start_time = f["analogs"][chn_key]["start_time"].value
                        chnData = f["analogs"][chn_key]["data"].value
                        resultData = self.__autoFilterThisChn(chnData,chnInfo)
                        del f["analogs"][chn_key]["data"]
                        f["analogs"][chn_key]["data"] = resultData
                    f.flush()

            delattr(self,"selectChan")


    def __selectChan(self):
        self.__cleanPk2()
        self.__cleanPk0()
        if self.chnList.currentText():
            self.saveChannelCheck.setChecked(False)
            self.FilterChannelCheck.setChecked(False)
            self.selectChan = self.chnList.currentText()
            print("Analog channel {0} is selected.".format(self.selectChan))
            self.sampling_rate,self.start_time,self.chnData = self.__load_chn_data(self.selectChan,self.file_name)

            if self.selectChan in self.chnResultPool.keys():
                self.FilterChannelCheck.setChecked(True)
                self.saveChannelCheck.setChecked(True)
                self.chnData = self.__autoFilterThisChn(self.chnData,self.chnResultPool[self.selectChan])

                self.__draw_pk2()
                self.__update_pk2_roi()

            if self.sampling_rate is not None:
                self.__draw_pk2()
                self.__addROIPk2()
                self.__update_pk2_roi()

    def __load_chn_data(self,selectChan,file_name):
        with hp.File(file_name,"r") as f:
            chn_key = "analog_{0}".format(selectChan)
            sampling_rate = f["analogs"][chn_key]["sampling_rate"].value
            start_time = f["analogs"][chn_key]["start_time"].value
            chnData = f["analogs"][chn_key]["data"].value
            if chnData.size:
                return sampling_rate,start_time,chnData
            else:
                return None,None,None

    def __saveThisChannelCheck(self):
        if self.saveChannelCheck.isChecked():
            algorithm = self.algorithmList.currentText()
            if algorithm !="addNew":
                ftype = self.ftypeWgt.currentText()
                self.chnResultPool[self.selectChan] = {"fs":self.sampling_rate,
                                                        "zerophase":eval(self.zerophaseWgt.currentText()),
                                                        "order":int(self.orderWgtt.text()),
                                                        "algorithm":algorithm,
                                                        "ftype":ftype}
            else:
                self.chnResultPool[self.selectChan] = {"algorithm":algorithm,
                                                        "fs":self.sampling_rate}

            if algorithm == "bandpass":
                if ftype in ["butter","bessel"]:
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                elif ftype == "cheby1":
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())     
                    rp = float(self.rpWgtt.text())            
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                elif ftype == "cheby2":
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())      
                    rs = float(self.rsWgtt.text())        
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rs"] = rs
                elif ftype == "ellip":
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())    
                    rp = float(self.rpWgtt.text())
                    rs = float(self.rsWgtt.text()) 
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                    self.chnResultPool[self.selectChan]["rs"] = rs
            elif algorithm == "bandstop":
                if ftype in ["butter","bessel"]:
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                elif ftype == "cheby1":
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())     
                    rp = float(self.rpWgtt.text())            
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                elif ftype == "cheby2":
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())      
                    rs = float(self.rsWgtt.text())        
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rs"] = rs
                elif ftype == "ellip":
                    lowcut = float(self.lowcutWgtt.text())
                    highcut = float(self.highcutWgtt.text())    
                    rp = float(self.rpWgtt.text())
                    rs = float(self.rsWgtt.text()) 
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                    self.chnResultPool[self.selectChan]["rs"] = rs             
            elif algorithm == "highpass":
                if ftype in ["butter","bessel"]:
                    lowcut = float(self.lowcutWgtt.text())
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                elif ftype == "cheby1":
                    lowcut = float(self.lowcutWgtt.text())
                    rp = float(self.rpWgtt.text())            
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                elif ftype == "cheby2":
                    lowcut = float(self.lowcutWgtt.text())
                    rs = float(self.rsWgtt.text())        
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["rs"] = rs
                elif ftype == "ellip":
                    lowcut = float(self.lowcutWgtt.text())
                    rp = float(self.rpWgtt.text())
                    rs = float(self.rsWgtt.text()) 
                    self.chnResultPool[self.selectChan]["lowcut"] = lowcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                    self.chnResultPool[self.selectChan]["rs"] = rs
            elif algorithm == "lowpass":
                if ftype in ["butter","bessel"]:
                    highcut = float(self.highcutWgtt.text())
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                elif ftype == "cheby1":
                    highcut = float(self.highcutWgtt.text())     
                    rp = float(self.rpWgtt.text())            
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                elif ftype == "cheby2":
                    highcut = float(self.highcutWgtt.text())      
                    rs = float(self.rsWgtt.text())        
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rs"] = rs
                elif ftype == "ellip":
                    highcut = float(self.highcutWgtt.text())    
                    rp = float(self.rpWgtt.text())
                    rs = float(self.rsWgtt.text()) 
                    self.chnResultPool[self.selectChan]["highcut"] = highcut
                    self.chnResultPool[self.selectChan]["rp"] = rp
                    self.chnResultPool[self.selectChan]["rs"] = rs
            
            if not self.FilterChannelCheck.isChecked():
                self.chnResultPool.pop(self.selectChan,None)
                self.saveChannelCheck.setChecked(False)
        else:
            self.chnResultPool.pop(self.selectChan,None)


    def __autoFilterThisChn(self,ite_data,chnInfo):
        if hasattr(self,"selectChan"):
            if chnInfo["algorithm"] !="addNew":
                fs = chnInfo["fs"]
                zerophase = chnInfo["zerophase"]
                order = chnInfo["order"]
                ftype = chnInfo["ftype"]
            else:
                filtered_data = self.myFilter.filter_design(ite_data,chnInfo["fs"])

            if chnInfo["algorithm"] == "bandpass":
                if ftype in ["butter","bessel"]:
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]
                    filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase)
                elif ftype == "cheby1":
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]  
                    rp = chnInfo["rp"]          
                    filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp)       
                elif ftype == "cheby2":
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]  
                    rs = chnInfo["rs"]          
                    filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rs=rs)        
                elif ftype == "ellip":
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]  
                    rp = chnInfo["rp"]          
                    rs = chnInfo["rs"]          
                    filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp,rs=rs) 
            elif chnInfo["algorithm"] == "bandstop":
                if ftype in ["butter","bessel"]:
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]  
                    filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase)
                elif ftype == "cheby1":
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]  
                    rp = chnInfo["rp"]      
                    filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp)          
                elif ftype == "cheby2":
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]  
                    rs = chnInfo["rs"]  
                    filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rs=rs)  
                elif ftype == "ellip":
                    lowcut = chnInfo["lowcut"]
                    highcut = chnInfo["highcut"]  
                    rp = chnInfo["rp"]
                    rs = chnInfo["rs"]
                    filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp,rs=rs)              
            elif chnInfo["algorithm"] == "highpass":
                if ftype in ["butter","bessel"]:
                    lowcut = chnInfo["lowcut"]
                    filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase)
                elif ftype == "cheby1":
                    lowcut = chnInfo["lowcut"]
                    rp = chnInfo["rp"]
                    filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase,rp=rp)
                elif ftype == "cheby2":
                    lowcut = chnInfo["lowcut"]
                    rs = chnInfo["rs"]   
                    filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase,rs=rs)
                elif ftype == "ellip":
                    lowcut = chnInfo["lowcut"]
                    rp = chnInfo["rp"]
                    rs = chnInfo["rs"]  
                    filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase,rp=rp,rs=rs)
            elif chnInfo["algorithm"] == "lowpass":
                if ftype in ["butter","bessel"]:
                    highcut = chnInfo["highcut"]
                    filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase)                         
                elif ftype == "cheby1":
                    highcut = chnInfo["highcut"] 
                    rp = chnInfo["rp"]   
                    filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase,rp=rp)                     
                elif ftype == "cheby2":
                    highcut = chnInfo["highcut"] 
                    rs = chnInfo["rs"] 
                    filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase,rs=rs)
                elif ftype == "ellip":
                    highcut = chnInfo["highcut"]
                    rp = chnInfo["rp"]
                    rs = chnInfo["rs"] 
                    filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase,rp=rp,rs=rs)                          
            return filtered_data

    def __FilterThisChnCheck(self):
        if hasattr(self,"selectChan"):
            self.sampling_rate,self.start_time,self.chnData = self.__load_chn_data(self.selectChan,self.file_name)
            ite_data = self.chnData
            fs = self.sampling_rate
            zerophase = eval(self.zerophaseWgt.currentText())
            order = int(self.orderWgtt.text())
            ftype = self.ftypeWgt.currentText()

            if self.FilterChannelCheck.isChecked():
                if self.algorithmList.currentText() == "bandpass":
                    if ftype in ["butter","bessel"]:
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())
                        filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase)
                    elif ftype == "cheby1":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())     
                        rp = float(self.rpWgtt.text())            
                        filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp)       
                    elif ftype == "cheby2":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())      
                        rs = float(self.rsWgtt.text())        
                        filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rs=rs)        
                    elif ftype == "ellip":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())    
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text()) 
                        filtered_data = self.__bandpass(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp,rs=rs) 
                    self.chnData = filtered_data
                elif self.algorithmList.currentText() == "bandstop":
                    if ftype in ["butter","bessel"]:
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())
                        filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase)
                    elif ftype == "cheby1":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())       
                        rp = float(self.rpWgtt.text())        
                        filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp)          
                    elif ftype == "cheby2":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())      
                        rs = float(self.rsWgtt.text())    
                        filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rs=rs)  
                    elif ftype == "ellip":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())  
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text())   
                        filtered_data = self.__bandstop(ite_data,lowcut,highcut,fs,order,ftype,zerophase,rp=rp,rs=rs)    
                    self.chnData = filtered_data          
                elif self.algorithmList.currentText() == "highpass":
                    if ftype in ["butter","bessel"]:
                        lowcut = float(self.lowcutWgtt.text())
                        filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase)
                    elif ftype == "cheby1":
                        lowcut = float(self.lowcutWgtt.text())
                        rp = float(self.rpWgtt.text())  
                        filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase,rp=rp)
                    elif ftype == "cheby2":
                        lowcut = float(self.lowcutWgtt.text())
                        rs = float(self.rsWgtt.text())     
                        filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase,rs=rs)
                    elif ftype == "ellip":
                        lowcut = float(self.lowcutWgtt.text())
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text())    
                        filtered_data = self.__highpass(ite_data,lowcut,fs,order,ftype,zerophase,rp=rp,rs=rs)
                    self.chnData = filtered_data
                elif self.algorithmList.currentText() == "lowpass":
                    if ftype in ["butter","bessel"]:
                        highcut = float(self.highcutWgtt.text())    
                        filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase)                         
                    elif ftype == "cheby1":
                        highcut = float(self.highcutWgtt.text())   
                        rp = float(self.rpWgtt.text())       
                        filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase,rp=rp)                     
                    elif ftype == "cheby2":
                        highcut = float(self.highcutWgtt.text())    
                        rs = float(self.rsWgtt.text())   
                        filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase,rs=rs)
                    elif ftype == "ellip":
                        highcut = float(self.highcutWgtt.text())   
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text())   
                        filtered_data = self.__lowpass(ite_data,highcut,fs,order,ftype,zerophase,rp=rp,rs=rs)  
                    self.chnData = filtered_data
                elif self.algorithmList.currentText()=="addNew":
                    if not hasattr(self,"i10"):
                        self.FilterChannelCheck.setChecked(False)
                    else:
                        filtered_data = self.myFilter.filter_design(self.chnData,self.sampling_rate)
                        self.chnData = filtered_data

                self.__draw_pk2()
                self.__update_pk2_roi()
            else:
                self.sampling_rate,self.start_time,self.chnData = self.__load_chn_data(self.selectChan,self.file_name)
                self.saveChannelCheck.setChecked(False)
                self.__saveThisChannelCheck()
                if self.sampling_rate is not None:
                    self.__draw_pk2()
                    self.__update_pk2_roi()


    def __bandpass(self,ite_data,lowcut,highcut,fs,order,ftype,zerophase,**args):
        fe = fs/2.0
        low = lowcut/fe
        high = highcut/fe
        if low<0:
            low=0
        if high>1:
            high=1

        if ftype == "cheby1":
            rp = args["rp"]
            z,p,k = signal.iirfilter(order,[low,high],btype="band",ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,[low,high],btype="band",ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = args["rp"]
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,[low,high],btype="band",ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,[low,high],btype="band",ftype=ftype,output="zpk")
        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data

    def __bandstop(self,ite_data,lowcut,highcut,fs,order,ftype,zerophase,**args):
        fe = fs/2.0
        low = lowcut/fe
        high = highcut/fe
        if low<0:
            low=0
        if high>1:
            high=1

        if ftype == "cheby1":
            rp = args["rp"]
            z,p,k = signal.iirfilter(order,[low,high],btype="bandstop",ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,[low,high],btype="bandstop",ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = args["rp"]
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,[low,high],btype="bandstop",ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,[low,high],btype="bandstop",ftype=ftype,output="zpk")

        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data

    def __highpass(self,ite_data,lowcut,fs,order,ftype,zerophase,**args):
        fe = fs/2.0
        low = lowcut/fe
        if low<0:
            low=0

        if ftype == "cheby1":
            rp = args["rp"]
            z,p,k = signal.iirfilter(order,low,btype="highpass",ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,low,btype="highpass",ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = args["rp"]
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,low,btype="highpass",ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,low,btype="highpass",ftype=ftype,output="zpk")
        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data

    def __lowpass(self,ite_data,highcut,fs,order,ftype,zerophase,**args):
        fe = fs/2.0
        high = highcut/fe
        if high>1:
            high=1

        if ftype == "cheby1":
            rp = args["rp"]
            z,p,k = signal.iirfilter(order,high,btype="lowpass",ftype=ftype,output="zpk",rp=rp)
        elif ftype == "cheby2":
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,high,btype="lowpass",ftype=ftype,output="zpk",rs=rs)
        elif ftype == "ellip":
            rp = args["rp"]
            rs = args["rs"]
            z,p,k = signal.iirfilter(order,high,btype="lowpass",ftype=ftype,output="zpk",rp=rp,rs=rs)
        else:
            z,p,k = signal.iirfilter(order,high,btype="lowpass",ftype=ftype,output="zpk")
        sos = signal.zpk2sos(z,p,k)
        ite_data = signal.sosfilt(sos,ite_data)
        if zerophase:
            ite_data = signal.sosfilt(sos,ite_data[::-1])[::-1]
        return ite_data

    def __useThisAlgorithm(self):
        self.algorithmArgTree.clear()
        self.saveChannelCheck.setChecked(False)
        self.FilterChannelCheck.setChecked(False)
        self.__algorithmInit()
        self.__FilterThisChnCheck()


    def __algorithmInit(self):
        if self.algorithmList.currentText() != "addNew":
            self.orderValue = 6
            self.lowcutValue = 0
            self.highcutValue = 100
            self.rpValue = 5
            self.rsValue = 40
            self.ftypeList = ["butter","cheby1","cheby2","ellip","bessel"]
            self.zerophaseList = ["True","False"]
            i7 = pg.TreeWidgetItem(["Default"])
            self.defaultbandpass = QtGui.QPushButton()
            self.defaultbandpass.clicked.connect(self.__defaultAlgorithm)
            self.defaultbandpass.setFixedWidth(60)
            self.defaultbandpass.setText("Click")
            i7.setWidget(1,self.defaultbandpass)
            self.algorithmArgTree.addTopLevelItem(i7)
            
            # order
            i3 = pg.TreeWidgetItem(["order"])
            self.orderWgtt = QtGui.QPushButton()
            self.orderWgtt.setText(str(self.orderValue))
            self.orderWgtt.setFixedWidth(60)
            self.orderWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
            self.orderWgtt.clicked.connect(self.__setorder)
            i3.setWidget(1,self.orderWgtt)
            self.algorithmArgTree.addTopLevelItem(i3)
            # ftype
            i4  = pg.TreeWidgetItem(["ftype"])
            self.ftypeWgt = QtGui.QComboBox()
            self.ftypeWgt.setFixedWidth(80)
            self.ftypeWgt.addItems(self.ftypeList)
            self.ftypeWgt.setCurrentIndex(0)
            self.ftypeWgt.currentIndexChanged.connect(self.__setftype)
            i4.setWidget(1, self.ftypeWgt)
            self.algorithmArgTree.addTopLevelItem(i4)
            # zerophase
            i5  = pg.TreeWidgetItem(["zerophase"])
            self.zerophaseWgt = QtGui.QComboBox()
            self.zerophaseWgt.setFixedWidth(80)
            self.zerophaseWgt.addItems(self.zerophaseList)
            self.zerophaseWgt.setCurrentIndex(0)
            self.zerophaseWgt.currentIndexChanged.connect(self.__setzerophase)
            i5.setWidget(1, self.zerophaseWgt)
            self.algorithmArgTree.addTopLevelItem(i5)
        else:
            i1 = pg.TreeWidgetItem(["upload"])
            self.uploadWgtt = QtGui.QPushButton()
            self.uploadWgtt.setText("Click")
            self.uploadWgtt.setFixedWidth(60)
            # self.lowcutWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
            self.uploadWgtt.clicked.connect(self.__upload)
            i1.setWidget(1,self.uploadWgtt)
            self.algorithmArgTree.addTopLevelItem(i1)

        if self.algorithmList.currentText() in ["bandpass","bandstop"]:
            # lowcut
            i1 = pg.TreeWidgetItem(["lowcut"])
            self.lowcutWgtt = QtGui.QPushButton()
            self.lowcutWgtt.setText(str(self.lowcutValue))
            self.lowcutWgtt.setFixedWidth(60)
            self.lowcutWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
            self.lowcutWgtt.clicked.connect(self.__setlowcut)
            i1.setWidget(1,self.lowcutWgtt)
            self.algorithmArgTree.addTopLevelItem(i1)
            # highcut
            i2 = pg.TreeWidgetItem(["highcut"])
            self.highcutWgtt = QtGui.QPushButton()
            self.highcutWgtt.setText(str(self.highcutValue))
            self.highcutWgtt.setFixedWidth(60)
            self.highcutWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
            self.highcutWgtt.clicked.connect(self.__sethighcut)
            i2.setWidget(1,self.highcutWgtt)
            self.algorithmArgTree.addTopLevelItem(i2)
        elif self.algorithmList.currentText()=="highpass":
            # lowcut
            i1 = pg.TreeWidgetItem(["lowcut"])
            self.lowcutWgtt = QtGui.QPushButton()
            self.lowcutWgtt.setText(str(self.lowcutValue))
            self.lowcutWgtt.setFixedWidth(60)
            self.lowcutWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
            self.lowcutWgtt.clicked.connect(self.__setlowcut)
            i1.setWidget(1,self.lowcutWgtt)
            self.algorithmArgTree.addTopLevelItem(i1)
        elif self.algorithmList.currentText()=="lowpass":
            # highcut
            i2 = pg.TreeWidgetItem(["highcut"])
            self.highcutWgtt = QtGui.QPushButton()
            self.highcutWgtt.setText(str(self.highcutValue))
            self.highcutWgtt.setFixedWidth(60)
            self.highcutWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
            self.highcutWgtt.clicked.connect(self.__sethighcut)
            i2.setWidget(1,self.highcutWgtt)
            self.algorithmArgTree.addTopLevelItem(i2)
            
    def __defaultAlgorithm(self):
        self.algorithmList.setCurrentIndex(0)
        self.ftypeWgt.setCurrentIndex(0)
        self.lowcutWgtt.setText(str(self.lowcutValue))
        self.highcutWgtt.setText(str(self.highcutValue))
        self.orderWgtt.setText(str(self.orderValue))
        self.zerophaseWgt.setCurrentIndex(0)

    def __setorder(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Set order.\nThis value should be a positive integer.\nDefault is {0}.'.format(self.orderValue))
        w.textValueSelected.connect(self.__setorder_value)
        w.exec_()
    def __setorder_value(self,value):
        if ftl.is_num(value) and (int(value)>=0) and (int(value) != int(self.highcutWgtt.text())):
            self.orderWgtt.setText(str(int(value)))
            self.__autoParamChanged()

    def __setftype(self):
        self.__ftype_args()
        self.__autoParamChanged()
    def __ftype_args(self):
        if hasattr(self,"i8"):
            self.algorithmArgTree.removeTopLevelItem(self.i8)
            delattr(self,"i8")
            delattr(self,"rpWgtt")
        if hasattr(self,"i9"):
            self.algorithmArgTree.removeTopLevelItem(self.i9)
            delattr(self,"i9")
            delattr(self,"rsWgtt")
        if self.ftypeWgt.currentText() in ["cheby1","ellip"]:
            if not hasattr(self,"i8"):
                self.i8 = pg.TreeWidgetItem(["rp"])
                self.rpWgtt = QtGui.QPushButton()
                self.rpWgtt.setText(str(self.rpValue))
                self.rpWgtt.setFixedWidth(60)
                self.rpWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
                self.rpWgtt.clicked.connect(self.__setrp)
                self.i8.setWidget(1,self.rpWgtt)
                self.algorithmArgTree.addTopLevelItem(self.i8)
        
        if self.ftypeWgt.currentText() in ["cheby2","ellip"]:
            if not hasattr(self,"i9"):
                self.i9 = pg.TreeWidgetItem(["rs"])
                self.rsWgtt = QtGui.QPushButton()
                self.rsWgtt.setText(str(self.rsValue))
                self.rsWgtt.setFixedWidth(60)
                self.rsWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
                self.rsWgtt.clicked.connect(self.__setrs)
                self.i9.setWidget(1,self.rsWgtt)
                self.algorithmArgTree.addTopLevelItem(self.i9)

    def __setrp(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Set rp value (float).\nThe maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number.\nDefault is {0}.'.format(self.rpValue))
        w.textValueSelected.connect(self.__setrp_value)
        w.exec_()
    def __setrp_value(self,value):
        if ftl.is_num(value) and (float(value)>=0) and (float(value) != float(self.rpWgtt.text())):
            self.rpWgtt.setText(value)
            self.__autoParamChanged()

    def __setrs(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Set rs value (float).\nThe minimum attenuation required in the stop band. Specified in decibels, as a positive number.\nDefault is {0}.'.format(self.rsValue))
        w.textValueSelected.connect(self.__setrs_value)
        w.exec_()
    def __setrs_value(self,value):
        if ftl.is_num(value) and (float(value)>=0) and (float(value) != float(self.rsWgtt.text())):
            self.rsWgtt.setText(value)
            self.__autoParamChanged()

    def __setzerophase(self):
        self.__autoParamChanged()

    def __setlowcut(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Set lowcut value.\nlowcut>=0\nDefault is {0}.'.format(self.lowcutValue))
        w.textValueSelected.connect(self.__setlowcut_value)
        w.exec_()
    def __setlowcut_value(self,value):
        if ftl.is_num(value) and (float(value)>=0) and (float(value) != float(self.lowcutWgtt.text())):
            self.lowcutWgtt.setText(value)
            self.__autoParamChanged()
    def __sethighcut(self):
        w=QtWidgets.QInputDialog()
        w.setLabelText('Set highcut value.\nlowcut<=sampling frequency (fs)\nDefault is {0}.'.format(self.highcutValue))
        w.textValueSelected.connect(self.__sethighcut_value)
        w.exec_()
    def __sethighcut_value(self,value):
        if ftl.is_num(value) and (float(value)>=0) and (float(value) != float(self.highcutWgtt.text())):
            self.highcutWgtt.setText(value)
            self.__autoParamChanged()

    def __autoParamChanged(self):
        self.saveChannelCheck.setChecked(False)
        self.FilterChannelCheck.setChecked(False)
        self.__FilterThisChnCheck()
        

    def __load(self):
        if hasattr(self,"chnResultPool") and (len(self.chnResultPool)>0):
            print("do you want to discard result")
        else:
            # Load a file-selection dialog
            name = QtGui.QFileDialog.getOpenFileName(filter="h5 (*.h5)")
            file_name = name[0]
            if len(file_name) > 0:
                self.file_name = file_name
                if hasattr(self,'selectChan'):
                    delattr(self,'selectChan')
                self.chnList.clear()
                self.chns = []
                with hp.File(self.file_name,'r') as f:
                    for key in f.keys():
                        if key=="analogs":
                            for chn in f["analogs"].keys():
                                self.chns.append(chn.split('_')[1])
                if self.chns:
                    self.saveChannelCheck.setChecked(False)
                    self.FilterChannelCheck.setChecked(False)
                    self.chnResultPool = dict()
                    self.chns = list(set(self.chns))
                    self.chns = [int(ite) for ite in self.chns]
                    self.chns.sort()
                    self.chns = [str(ite) for ite in self.chns]
                    self.chnList.addItems(self.chns)
                else:
                    print("There is no analog channels in this file")


    def __reset(self):
        if hasattr(self,"selectChan"):
            self.FilterChannelCheck.setChecked(False)
            self.saveChannelCheck.setChecked(False)
            self.chnResultPool.clear()
            self.chnList.clear()
            self.chnList.addItems(self.chns)

    def __save(self):
        if hasattr(self,"chns"):
            chnNotProcessed = set(self.chns).difference(set(self.chnResultPool.keys()))
            chnNotProcessed = sorted([int(ite) for ite in chnNotProcessed])
            self.unProcessedChns = copy.copy(chnNotProcessed)
            self.unProcessedChns = [str(ite) for ite in self.unProcessedChns]
            if len(chnNotProcessed)>0:
                err_msg = QtWidgets.QMessageBox()
                err_msg.setIcon(QtWidgets.QMessageBox.Information)
                err_msg.setText("Data in these channels are not processed: \n{0}. \n \n[save + autoSortUnprocessedChns]:  results of processed channels will be saved, and unprocessed channels will be auto processed using the algorithm with setted parameters in the current main control panel.\n\n[onlySaveProcessedChns]:  only results of processed channels are saved, and data in unprocessed channels are unchanged.".format(chnNotProcessed))
                
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
                    chn_key = "analog_{0}".format(selectChan)
                    sampling_rate = f["analogs"][chn_key]["sampling_rate"].value
                    start_time = f["analogs"][chn_key]["start_time"].value
                    chnData = f["analogs"][chn_key]["data"].value
                    resultData = self.__autoFilterThisChn(chnData,self.chnResultPool[selectChan])
                    del f["analogs"][chn_key]["data"]
                    f["analogs"][chn_key]["data"] = resultData
                f.flush()
            self.chnResultPool.clear()
            self.saveChannelCheck.setChecked(False)
            self.FilterChannelCheck.setChecked(False)
        elif msg.text()=="save + autoSortUnprocessedChns":
            # save processed chn first
            with hp.File(self.file_name,"a") as f:
                for selectChan in self.chnResultPool.keys():
                    chn_key = "analog_{0}".format(selectChan)
                    sampling_rate = f["analogs"][chn_key]["sampling_rate"].value
                    start_time = f["analogs"][chn_key]["start_time"].value
                    chnData = f["analogs"][chn_key]["data"].value
                    resultData = self.__autoFilterThisChn(chnData,self.chnResultPool[selectChan])
                    del f["analogs"][chn_key]["data"]
                    f["analogs"][chn_key]["data"] = resultData
                f.flush()
            # processed unprocessed channel and save 
            algorithm = self.algorithmList.currentText()
            if algorithm == "addNew":
                with hp.File(self.file_name,"a") as f:
                    for selectChan in self.unProcessedChns:
                        if hasattr(self,"i10"):
                            chn_key = "analog_{0}".format(selectChan)
                            sampling_rate = f["analogs"][chn_key]["sampling_rate"].value
                            chnInfo = {"algorithm":"addNew","fs":sampling_rate}
                            start_time = f["analogs"][chn_key]["start_time"].value
                            chnData = f["analogs"][chn_key]["data"].value
                            resultData = self.__autoFilterThisChn(chnData,chnInfo)
                            del f["analogs"][chn_key]["data"]
                            f["analogs"][chn_key]["data"] = resultData
                    f.flush()
            else:
                ftype = self.ftypeWgt.currentText()
                chnInfo = {
                            #"fs":self.sampling_rate,
                            "zerophase":eval(self.zerophaseWgt.currentText()),
                            "order":int(self.orderWgtt.text()),
                            "algorithm":algorithm,
                            "ftype":ftype}
                if algorithm == "bandpass":
                    if ftype in ["butter","bessel"]:
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                    elif ftype == "cheby1":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())     
                        rp = float(self.rpWgtt.text())            
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                        chnInfo["rp"] = rp
                    elif ftype == "cheby2":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())      
                        rs = float(self.rsWgtt.text())        
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                        chnInfo["rs"] = rs
                    elif ftype == "ellip":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())    
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text()) 
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                        chnInfo["rp"] = rp
                        chnInfo["rs"] = rs
                elif algorithm == "bandstop":
                    if ftype in ["butter","bessel"]:
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                    elif ftype == "cheby1":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())     
                        rp = float(self.rpWgtt.text())            
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                        chnInfo["rp"] = rp
                    elif ftype == "cheby2":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())      
                        rs = float(self.rsWgtt.text())        
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                        chnInfo["rs"] = rs
                    elif ftype == "ellip":
                        lowcut = float(self.lowcutWgtt.text())
                        highcut = float(self.highcutWgtt.text())    
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text()) 
                        chnInfo["lowcut"] = lowcut
                        chnInfo["highcut"] = highcut
                        chnInfo["rp"] = rp
                        chnInfo["rs"] = rs             
                elif algorithm == "highpass":
                    if ftype in ["butter","bessel"]:
                        lowcut = float(self.lowcutWgtt.text())
                        chnInfo["lowcut"] = lowcut
                    elif ftype == "cheby1":
                        lowcut = float(self.lowcutWgtt.text())
                        rp = float(self.rpWgtt.text())            
                        chnInfo["lowcut"] = lowcut
                        chnInfo["rp"] = rp
                    elif ftype == "cheby2":
                        lowcut = float(self.lowcutWgtt.text())
                        rs = float(self.rsWgtt.text())        
                        chnInfo["lowcut"] = lowcut
                        chnInfo["rs"] = rs
                    elif ftype == "ellip":
                        lowcut = float(self.lowcutWgtt.text())
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text()) 
                        chnInfo["lowcut"] = lowcut
                        chnInfo["rp"] = rp
                        chnInfo["rs"] = rs
                elif algorithm == "lowpass":
                    if ftype in ["butter","bessel"]:
                        highcut = float(self.highcutWgtt.text())
                        chnInfo["highcut"] = highcut
                    elif ftype == "cheby1":
                        highcut = float(self.highcutWgtt.text())     
                        rp = float(self.rpWgtt.text())            
                        chnInfo["highcut"] = highcut
                        chnInfo["rp"] = rp
                    elif ftype == "cheby2":
                        highcut = float(self.highcutWgtt.text())      
                        rs = float(self.rsWgtt.text())        
                        chnInfo["highcut"] = highcut
                        chnInfo["rs"] = rs
                    elif ftype == "ellip":
                        highcut = float(self.highcutWgtt.text())    
                        rp = float(self.rpWgtt.text())
                        rs = float(self.rsWgtt.text()) 
                        chnInfo["highcut"] = highcut
                        chnInfo["rp"] = rp
                        chnInfo["rs"] = rs
                with hp.File(self.file_name,"a") as f:
                    for selectChan in self.unProcessedChns:
                        chn_key = "analog_{0}".format(selectChan)
                        sampling_rate = f["analogs"][chn_key]["sampling_rate"].value
                        chnInfo["fs"] = sampling_rate
                        start_time = f["analogs"][chn_key]["start_time"].value
                        chnData = f["analogs"][chn_key]["data"].value
                        resultData = self.__autoFilterThisChn(chnData,chnInfo)
                        del f["analogs"][chn_key]["data"]
                        f["analogs"][chn_key]["data"] = resultData
                    f.flush()
            self.chnResultPool.clear()
            self.saveChannelCheck.setChecked(False)
            self.FilterChannelCheck.setChecked(False)
        print("save finished.")

    def __draw_pk2(self):
        self.__cleanPk2()
        times = float(self.start_time) + np.arange(self.chnData.shape[0])/(self.sampling_rate/1000.0)
        self.sigLineItem = MultiLine(np.array([times]),np.array([self.chnData]),"g")
        self.pk2.addItem(self.sigLineItem)

    def __update_pk2_roi(self):
        self.minX,self.maxX = self.pk2_roi.getRegion()
        idx_0 = (self.minX - self.start_time)*(self.sampling_rate/1000.0)
        idx_1 = (self.maxX - self.start_time)*(self.sampling_rate/1000.0)
        if idx_0<0:
            idx_0 = 0
        if idx_1<0:
            idx_1 = 0
        idx_0 = int(idx_0)
        idx_1 = int(idx_1)
        self.sig_pk0 = self.chnData[idx_0:idx_1]
        self.__draw_pk0()

    def __draw_pk0(self):
        self.__cleanPk0()
        if self.sig_pk0.size:
            times = np.linspace(self.minX,self.maxX,self.sig_pk0.shape[0],endpoint=False)
            self.partSigLineItem = MultiLine(np.array([times]),np.array([self.sig_pk0]),"g")
            self.pk0.addItem(self.partSigLineItem)

    def __cleanPk0(self):
        if hasattr(self,"partSigLineItem"):
            self.pk0.removeItem(self.partSigLineItem)
            delattr(self,"partSigLineItem")
    def __cleanPk2(self):
        if hasattr(self,"sigLineItem"):
            self.pk2.removeItem(self.sigLineItem)
            delattr(self,"sigLineItem")

    def __addROIPk2(self):
        if hasattr(self,"pk2_roi"):
            self.pk2.removeItem(self.pk2_roi)
            delattr(self,"pk2_roi")
        self.pk2_roi = pg.LinearRegionItem()
        self.pk2_roi.setRegion([self.start_time,self.start_time+5000])
        self.pk2_roi.sigRegionChanged.connect(self.__update_pk2_roi)
        self.pk2.addItem(self.pk2_roi,ignoreBounds=True)
    def __upload(self):
        err_msg = QtWidgets.QMessageBox()
        err_msg.setIcon(QtWidgets.QMessageBox.Information)
        err_msg.setText("upload .py file for signal filtering.\nThe .py file must contain only one function, named filter_design.\nThe function structure should be like this:\n\ndef filter_design(input_data,fs):\n       ...\n       ...\n       return result_data")
        
        # autoBtn = QtGui.QPushButton()
        # autoBtn.setText("autoSort unprocessed chns")
        err_msg.addButton("Yes",QtGui.QMessageBox.YesRole)
        err_msg.addButton("Cancel",QtGui.QMessageBox.NoRole)                
        err_msg.buttonClicked.connect(self.__checkUpload)
        err_msg.exec_()

    def __checkUpload(self,msg):   
        if msg.text()=="Yes":
            name = QtGui.QFileDialog.getOpenFileName(filter=".py (*.py)")
            file_name = name[0]
            if len(file_name) > 0:
                if hasattr(self,"i10"):
                    self.algorithmArgTree.removeTopLevelItem(self.i10)
                    delattr(self,"i10")
                    delattr(self,"myFilter")
                self.myFilter = imp.load_source("filter_design",file_name)
                self.i10 = pg.TreeWidgetItem([""])
                myFilterWgtt = QtGui.QPushButton()
                myFilterWgtt.setText("myFilter")
                myFilterWgtt.setFixedWidth(60)
                myFilterWgtt.setStyleSheet("background-color: rgb(190,190,190);border-radius: 6px;")
                # rpWgtt.clicked.connect(self.__setrp)
                self.i10.setWidget(1,myFilterWgtt)
                self.algorithmArgTree.addTopLevelItem(self.i10)

        elif msg.text()=="Cancel":
            pass

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
