import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
relative_path = os.path.join(current_dir, '../')
sys.path.append(relative_path)

import time
import numpy as np
import cv2
from utilsTraitement import listFilesDirectory,halfHeightCrop,readYamlConfig,getTrainingStatus
from models.trainDistillation import AnomalyDistillation
from datasetCreationModif import basicMotionDetection
from decision import decision
from decisionPrecise import decisionPrecise



"""
The class for video processing
"""
class processingVideo():
    def __init__(self,data,thresholds,mode="reduced",DG=False):
        self.data=data
        self.lastIdEvent =1
        self.dbHost=data['Database']['dbHost']
        self.dbName = data['Database']['localdbName']
        self.username = data['Database']['localusrName']
        self.password = data['Database']['localusrPwd']

        self.obj = data['Models']['smallModel']
        self.threshold = float(thresholds[str(self.obj)])
        self.processType = data['Local']['processType']
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
    Function to process a video.

    Args:
        video_path: path of the video
    Returns:
        None
    """

    def processVideo(self,video_path):
        cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG) # 
        print(video_path)
        if not cap.isOpened():
            print("Error when opening the video") 
            
        compteurFrame=0
        previous_frame = np.zeros([1, 1])
        while (cap.isOpened()):
            ret, self.frame = cap.read()
            compteurFrame=compteurFrame+1
            if ret == True and compteurFrame > 2: 
                mvtDetected = basicMotionDetection(self.frame, previous_frame)
            previous_frame = self.frame
            if ret == True and compteurFrame>2 and mvtDetected:  
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
                    print("Too many defects detected, defect confirmed without precise decision")          
                timeAfterFeatures = time.perf_counter()
                print("fullProcess : " + str(timeAfterFeatures - timeBefore)) 
                if (self.DG):
                    status=getTrainingStatus(self.dbHost,self.dbName,self.username,self.password,idCam=self.idCam)
                    if (status=="test"):
                        break
                             
            if ret!=True:
                break
        cap.release()
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    print("============= TRAITEMENT VIDEO =============")

    config_file = "config.yaml"
    thresholds_file = "thresholds.yaml"
    
    data = readYamlConfig(config_file)
    thresholds=readYamlConfig(thresholds_file)
    
    process = processingVideo(data,thresholds,mode="reduced")
    # crop image en deux (on fera cela côté logiciel plutôt à l'avenir
    process.half = data['Config']['half']
    if (process.processType=='video'):
        video_path=data['Local']['video_path']
        process.processVideo(video_path)
    if (process.processType=='directory'):
        directory=data['Local']['video_path']
        videoFiles, fileNames=listFilesDirectory(directory)
        compteur=0
        for file in videoFiles:        
            process.videoName=fileNames[compteur] #on récupere le nom de la video pour l'enregistrement des images
            compteur=compteur+1
            print("PROCESSING VIDEO : "+str(compteur)+"/"+str(len(videoFiles)))
            process.resetParams() #reset les valeurs à chaque changement de vidéo
            process.processVideo(file)
    exit()
