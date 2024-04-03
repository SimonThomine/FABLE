import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
relative_path = os.path.join(current_dir, '../')
sys.path.append(relative_path)

import time
import numpy as np
import cv2
from utilsTraitement import halfHeightCrop,readYamlConfig,getTrainingStatus
from models.trainDistillation import AnomalyDistillation
from datasetCreationModif import basicMotionDetection
from decision import decision
from decisionPrecise import decisionPrecise

from harvesters.core import Harvester
import logging

logging.basicConfig(filename='logTest.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

"""
The class for processing via a GenICam-compatible camera
"""
class processingGenicam():
    def __init__(self,data,thresholds,mode="reduced",DG=False):
        self.data=data
        self.lastIdEvent =1
        self.dbHost=data['Database']['dbHost']
        self.dbName = data['Database']['localdbName']
        self.username = data['Database']['localusrName']
        self.password = data['Database']['localusrPwd']
        
        self.obj = data['Models']['smallModel']
        self.threshold = float(thresholds[str(self.obj)])
        self.idCam=data['Config']['idCam']
        self.DG=DG
        self.mode=mode
        self.videoName=""
        
        self.imageSize=data['Config']['imageSize']
        
        self.fastProcess=data['Config']['fastProcess']
        
        #Load small model
        self.smallKD=AnomalyDistillation(data,inference=True,big=False,modelName=data['Models']['smallModelName'],out_indices=[],DG=DG)
        # Load big model
        self.kd=AnomalyDistillation(data,inference=True,big=True,modelName=data['Models']['bigModelName'],out_indices=[],DG=DG)
        
        self.decision=decision(data,self.threshold)

        self.decisionPrecise=decisionPrecise(data)

        self.frame=[]
        self.drawing=[]
        self.half=data['Config']['half']     
        
    """
    Function to reset decision parameters in case of a discontinuity with motion detection.

    Args:
        None
    Returns:
        None
    """

    def resetParams(self):
        self.decision=decision(self.data,self.threshold)
        self.decisionPrecise=decisionPrecise(self.data)    
    
    
    """
    Function to perform image retrieval from the camera and processing.

    Args:
        data: YAML file

    Returns:
        None
    """

    def processGenicam(self,data):
        h = Harvester()
        h.add_file(data['Genicam']['cti'])
        h.update()
        ia = h.create()
        ia.remote_device.node_map.PixelFormat.value = 'BayerRG8'
        ia.remote_device.node_map.ExposureTime.set_value(int(data['Genicam']['exposure'])) 
        ia.start()
        logging.warning("camera harvester init")
        compteurFrame=0
        previous_frame = np.zeros([1, 1])
        
        while (True):
            compteurFrame=compteurFrame+1
            with ia.fetch() as buffer:
                component = buffer.payload.components[0]
                _2d = component.data.reshape(int(data['Genicam']['height']), int(data['Genicam']['width']))
                self.frame = _2d
                if 'Bayer' in ia.remote_device.node_map.PixelFormat.value:
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BayerRG2RGB)  
            if compteurFrame > 2:
                mvtDetected = basicMotionDetection(self.frame, previous_frame)
            previous_frame = self.frame
            if  compteurFrame>2 and mvtDetected:
                if self.half == True:         
                    self.frame=halfHeightCrop(self.frame)
                self.highResFrame=np.copy(self.frame)
                self.frame=cv2.resize(self.frame, (int(self.frame.shape[1] / 2), int(self.frame.shape[0] / 2))) 
                timeBefore = time.perf_counter()
                self.drawing=np.copy(self.frame)   
                defectiveIndexes,defectiveMasks,defectiveCoords,id_event=self.decision.inferenceForDecision(self.frame,self.kd,self.drawing)  
                if (len(defectiveIndexes)<4):
                    self.decisionPrecise.inferenceForPreciseDecision(self.highResFrame,defectiveIndexes,defectiveMasks,defectiveCoords,self.smallKD,id_event,full_frame=self.drawing)
                elif(len(defectiveIndexes)>4):
                    logging.warning("Too many defects detected, defect confirmed without precise decision")
                timeAfterFeatures = time.perf_counter()
                logging.warning("fullProcess : " + str(timeAfterFeatures - timeBefore))
                print("fullProcess : " + str(timeAfterFeatures - timeBefore)) 
                if (self.DG):
                    status=getTrainingStatus(self.dbHost,self.dbName,self.username,self.password,idCam=self.idCam)
                    if (status=="test"):
                        break              
    

if __name__ == '__main__':
    print("============= TRAITEMENT GENICAM ============= ")

    config_file = "config.yaml"
    thresholds_file = "thresholds.yaml"
    
    data = readYamlConfig(config_file)
    thresholds=readYamlConfig(thresholds_file)
    
    process = processingGenicam(data,thresholds,mode="reduced")
    process.half = data['Config']['half']
    process.resetParams() 
    process.processGenicam(data)   
    exit()
