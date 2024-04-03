import numpy as np
from skimage import morphology
import cv2
import logging
from utilsTraitement import insertEventSql,insertPatchSql,saveImagesDefault,normalizeBatch,createListPatchRecouvrement,cropAnoMap, enhanceContrastImage,extractNMaxValues
from utils.functions import cal_anomaly_maps
from utils.util import denormalization
from datetime import datetime
import os
import torch

logging.basicConfig(filename='logTest.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class decision():
    def __init__(self,data,threshold):
        self.listEcartMoyen=[]     
        self.listCropMeanTemporal=[[]]
        self.listEcartMoyenHeight=[]
        
        self.threshold=threshold
        self.diffTemporalNorm=0
        self.diffMeanNorm=0
        self.ecartMoyMax=0
        
        self.dbHost=data['Database']['dbHost']
        self.dbName=data['Database']['localdbName']
        self.username=data['Database']['localusrName']
        self.password=data['Database']['localusrPwd']
        self.imgSize=data['Config']['imageSize']
        
        self.isCrop=data['Config']['cropAnoMapDet']
        
        self.detThresh=data['Config']['detThresh']
        self.fast=data['Config']['fastProcess']
        self.overlap=data['Config']['overlap']
        self.halfPrecision=data['Models']['halfPrecision']
    """
    The function that performs inference and decision-making process.
    It aggregates other functions of the class.
    It returns values for the precise decision class.

    Args:
        frame: image to process
        kd: anomaly detection model
        drawing: full-frame image that will be saved if necessary
        
    Returns:
        defectiveIndexes: list of indexes of patches with defects
        defectiveMasks: list of masks of patches with defects
        defectiveCoords: list of coordinates of patches with defects
        id_event: id of the recorded event (full frame)
    """
   
    def inferenceForDecision(self,frame,kd,drawing):
        listImage = []
        listCrop = []
        listCoord=[]
        scores = []      
        _,patchNbrHeight=createListPatchRecouvrement(frame, listImage, listCrop, listCoord,imgSize=self.imgSize,overlap=self.overlap,fast=self.fast)
        batch = torch.stack(listImage).to('cuda')
        batch=normalizeBatch(batch) .half() if self.halfPrecision else normalizeBatch(batch) #! Ajout half   
        with torch.set_grad_enabled(False):     
            features_t =kd.model_t(batch)
            features_s = kd.model_s(batch)         
            scores = cal_anomaly_maps(features_s, features_t, self.imgSize)
        if self.isCrop:
            scores=cropAnoMap(scores)

        scores = np.asarray(scores)       
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        N_img_scores=extractNMaxValues(scores)
        
        defectiveIndexes,defectiveMasks,defectiveCoords,id_event=self.analyzePatches(img_scores,N_img_scores,scores,listCrop,listCoord,patchNbrHeight,drawing)   
        return defectiveIndexes,defectiveMasks,defectiveCoords,id_event
    
    
    """
    Function for decision-making based on previous scores and current scores.

    Args:
        img_scores: list of scores of patches
        i: index of the patch to process
        nbPatchHeight: number of patches aligned vertically
        
    Returns:
        True if the patch is considered defective, False otherwise
    """  
    def finalDecisionRemastered(self,img_scores,i,nbPatchHeight=4):
        if(len(self.listCropMeanTemporal[0])==0): 
            for j in range(len(img_scores)):
                self.listCropMeanTemporal.append([])
                self.listEcartMoyen.append(0)
                               
        if (len(self.listCropMeanTemporal[0])<3): 
            cropIndexWidth=i//nbPatchHeight
            gapCrop=nbPatchHeight*cropIndexWidth
            self.listEcartMoyen[i]=img_scores[i]-(sum(img_scores[gapCrop:gapCrop+nbPatchHeight]) / nbPatchHeight)
            self.listCropMeanTemporal[i].append(img_scores[i])
            return False  
        
        
      
        ecartMoyTemp=(sum(self.listCropMeanTemporal[i]) / len(self.listCropMeanTemporal[i]))
        self.diffTemporalNorm=(img_scores[i]-ecartMoyTemp)/ecartMoyTemp
        self.diffMeanNorm=((img_scores[i]-(sum(img_scores) / len(img_scores)))-self.listEcartMoyen[i])/self.listEcartMoyen[i]
        
        if ( self.diffTemporalNorm>self.detThresh):
            return True
        
        self.listEcartMoyen[i] = img_scores[i] - (sum(img_scores) / len(img_scores))
        if (len(self.listCropMeanTemporal[i])<5):
            self.listCropMeanTemporal[i].append(img_scores[i])
        else:
            self.listCropMeanTemporal[i].pop(0)
            self.listCropMeanTemporal[i].append(img_scores[i])
        return False
    
    """
    Function for decision-making based on the patch mask.

    Args:
        scores: list of anomaly maps of patches
        i: index of the patch to process
        
    Returns:
        True if the patch is considered defective, False otherwise
        mask: mask of the patch
    """
    def verifCropAnomaly(self,scores,i):
        score = scores[i]*1000      
        min_value = np.min(score)
        max_value = np.max(score) 

        self.ecartMoyMax=max_value-min_value

        mask = scores[i] * 1000

        mask = (mask - np.min(mask)) /(np.max(mask) - np.min(mask))

        mask[mask > 0.7] = 1 
        mask[mask <= 0.7] = 0
        radius = 4
        kernel = morphology.disk(radius)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        nbWhite=cv2.countNonZero(mask)

        if (nbWhite==0): 
            return False,[]                
        return True,mask
    
    def sizeDecision(self,N_img_scores,i):
        max_value=np.max(N_img_scores[i])
        min_value=np.min(N_img_scores[i])
        ecart=(max_value-min_value)/max_value
        if (ecart<0.1):
            print("big defect confirmed")
            return True
        else : 
            return False
        
    
    """
    Function that aggregates the two previous functions for decision-making
    and processes all patches, then saves images if necessary.

    Args:
        img_scores: list of scores of patches
        scores: list of anomaly maps of patches
        listCrop: list of patches
        listCoord: list of coordinates of patches
        patchNbrHeight: number of patches aligned vertically
        drawing: full-frame image that will be saved if necessary
        
    Returns:
        defectiveIndexes: list of indexes of patches with defects
        defectiveMasks: list of masks of patches with defects
        defectiveCoords: list of coordinates of patches with defects
        id_event: id of the recorded event (full frame)
    """

    def analyzePatches(self,img_scores,N_img_scores,scores,listCrop,listCoord,patchNbrHeight,drawing):
        defectDetected=False
        id_event=0
        defectiveIndexes=[]
        defectiveMasks=[]
        defectiveCoords=[]
        for i in range(len(img_scores)): 
            if(self.finalDecisionRemastered(img_scores,i,patchNbrHeight)): 
                isAnomaly,mask=self.verifCropAnomaly(scores,i)    
                if(isAnomaly): 
                    if (not defectDetected):
                        logging.warning("DEFECT DETECTED")
                        print("DEFECT DETECTED")
                        id_event=self.saveFullFrame(drawing,"0x01893eecc7cdfeedd53a580dc329725a")
                    defectDetected=True
                      
                    defectiveIndexes.append(i)
                    defectiveMasks.append(mask)
                    defectiveCoords.append(listCoord[i]) 
                    mask=[]  
        return defectiveIndexes,defectiveMasks,defectiveCoords,id_event
    
    """
    Function for saving the full-frame image.

    Args:
        drawing: full-frame image to be saved
        idcamera: camera id
        
    Returns:
        id_event: id of the event recorded in the database
    """
    def saveFullFrame(self,drawing,idcamera):
        id_event=insertEventSql(self.dbHost,self.dbName, self.username, self.password,idcamera)
        now = datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")
        hour = now.strftime("%H")
        path = "data/Images/" + str(year) + "/" + str(month) + "/" + str(day) + "/" + str(hour) + "/"
        os.makedirs(path, exist_ok=True)
        drawing=enhanceContrastImage(drawing)
        cv2.imwrite(path +"Full_"+ str(id_event)+ ".jpg", drawing)
        return id_event
    
    """
    Function for saving a defect crop and its mask.

    Args:
        listCrop: list of patches
        mask: mask of the patch
        img_scores: list of scores of patches
        i: index of the patch to save
        listCoord: list of coordinates of patches
        id_event: id of the event recorded (full frame)

    Returns:
        None
    """          
    def saveImage(self,listCrop,mask,img_scores,i,listCoord,id_event):
        img = np.asarray(listCrop[i])
        img = np.transpose(img, (2, 0, 1))
        img = denormalization(img)  
        id_patch=insertPatchSql(self.dbHost,self.dbName, self.username, self.password, id_event, img_scores[i],listCoord[i],precise=False)
        saveImagesDefault(listCrop[i], mask, 1, id_patch)
