import numpy as np
from skimage import morphology
import logging
from utilsTraitement import insertPatchSql,saveImagesDefault,normalizeBatch,cropAnoMap,enhanceContrastImage
from utils.functions import cal_anomaly_maps
from utils.util import denormalization
from utilsTraitement import extractInterestPatches
import torch
import cv2
from datetime import datetime
import os

from scipy import ndimage

logging.basicConfig(filename='logTest.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class decisionPrecise():
    def __init__(self,data):    
        
        self.listCropHighRes=[[]]  
        self.listFrameHighRes=[]
        
        self.threshold=data['Config']['preciseDetThresh']
        self.imgSize=data['Config']['imageSize']
        
        self.dbHost=data['Database']['dbHost']
        self.dbName=data['Database']['localdbName']
        self.username=data['Database']['localusrName']
        self.password=data['Database']['localusrPwd']
        
        self.fast=data['Config']['fastProcess']
        self.overlap=data['Config']['overlap']

        self.isCrop=data['Config']['cropAnoMapPreciseDet']
        self.halfPrecision=data['Models']['halfPrecision']
        
        self.debugDrawing=data['Config']['debugDrawing']
        self.nbPastImages=5

    
    """
    The function that performs inference and precise decision-making process.
    It aggregates other functions of the class.
    It takes arguments from the decision class.

    Args:
        highResFrame: high-resolution image
        defectiveIndexList: list of indexes of detected defects
        defectiveMasks: list of masks of detected defects
        defectiveCoords: list of coordinates of detected defects
        smallKD: anomaly detection model
        id_event: id of the event recorded in the database

    Returns:
        None
    """

    def inferenceForPreciseDecision(self,highResFrame,defectiveIndexList,defectiveMasks,defectiveCoords,smallKD,id_event,full_frame):
        
        batch, cropsInterestPatchs, coordInterestPatchs=self.gatherAndWonder(highResFrame,defectiveIndexList,defectiveMasks,defectiveCoords)
        
        if len(cropsInterestPatchs)==0 or len(self.listFrameHighRes)<self.nbPastImages:
            return
        if (len(batch)>0):
            batch = torch.stack(batch).to('cuda')
            batch=normalizeBatch(batch).half() if self.halfPrecision else normalizeBatch(batch) #! Ajout half
            with torch.set_grad_enabled(False):     
                    features_t =smallKD.model_t(batch)
                    features_s = smallKD.model_s(batch)             
                    scores = cal_anomaly_maps(features_s, features_t, self.imgSize)
            
            if self.isCrop:
                scores=cropAnoMap(scores)
            scores = np.asarray(scores)       
            img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
            
            indexDefect,scoresDefect,imgScoresDefects=self.validateAndLocateDefect(scores,img_scores,len(cropsInterestPatchs))
            
            drawing=np.copy(full_frame)
            for k,index in enumerate(indexDefect):
                imageToSave=cropsInterestPatchs[index]
                coordToSave=coordInterestPatchs[index]
                score=scoresDefect[k]
                img_score=imgScoresDefects[k]
                self.saveImage(imageToSave,score,img_score,id_event,coord=coordToSave)
                if self.debugDrawing:
                    cv2.rectangle(drawing, (coordToSave[1]//2, coordToSave[0]//2), (coordToSave[1]//2+self.imgSize//2, coordToSave[0]//2+self.imgSize//2), (0, 0, 255), 2)
            if self.debugDrawing and len(indexDefect)>0:
                self.saveFullFrame(drawing,id_event)  
                
        
    
    """
    Function that retrieves the patches of interest and their associated coordinates.
    It also stores the entire images for the next decision-making if no defect is detected.

    Args:
        highResFrame: high-resolution image
        defectiveIndexes: list of indexes of detected defects
        defectiveMasks: list of masks of detected defects
        defectiveCoords: list of coordinates of detected defects
        
    Returns:
        batch: batch of patches of interest
        cropsInterestPatchs: list of patches of interest
        coordInterestPatchs: list of coordinates of patches of interest
    """

    def gatherAndWonder(self,highResFrame,defectiveIndexes,defectiveMasks,defectiveCoords):
        if (len(defectiveIndexes)==0):
            if len(self.listFrameHighRes)>=self.nbPastImages:
                self.listFrameHighRes.pop(0)
            self.listFrameHighRes.append(highResFrame)
            return [],[],[]
        else:
            coordInterestPatchs=self.extractDefectCoord(defectiveMasks,defectiveCoords,highResFrame)
            batch,cropsInterestPatchs=extractInterestPatches(highResFrame,coordInterestPatchs,self.listFrameHighRes,self.imgSize)

            return batch, cropsInterestPatchs, coordInterestPatchs
        
    """
    Function to extract the coordinates of the patches of interest.
    We base this on the decision mask and take the patch centered on the mask area.

    Args:
        defectiveMasks: list of masks of detected defects
        defectiveCoords: list of coordinates of detected defects
        highResFrame: high-resolution image
        
    Returns:
        coordInterestPatchs: list of coordinates of patches of interest
    """

    def extractDefectCoord(self,defectiveMasks,defectiveCoords,highResFrame):
        maxPossibleY=highResFrame.shape[0]-self.imgSize
        maxPossibleX=highResFrame.shape[1]-self.imgSize
        
        coordInterestPatchs=[]
        for i,mask in enumerate(defectiveMasks):
            labeled_array, num_features = ndimage.label(mask == 255)

            sizes = ndimage.sum(mask == 255, labeled_array, range(1, num_features + 1))
            pairs = [(valeur, indice+1) for indice, valeur in enumerate(sizes)]
            pairs_tries = sorted(pairs, key=lambda x: x[0], reverse=True)
            indices = [pair[1] for pair in pairs_tries]
            
            coordPatchsTemp=[]
            for label in indices:
                y, x = np.where(labeled_array == label)
                center=(np.mean(y), np.mean(x))
                yCoord=min(int(center[0])*2-self.imgSize//2 ,maxPossibleY) 
                yCoord=max(yCoord,0)
                xCoord=min(int(center[1])*2-self.imgSize//2 ,maxPossibleX)
                xCoord=max(xCoord,0)
                coordPatchsTemp.append((yCoord,xCoord))
            
            for coordPatch in coordPatchsTemp:
                if coordPatch[0]<self.imgSize//2 and coordPatch[1]<self.imgSize//2:
                    coordInterestPatchs.append((0+ defectiveCoords[i][1]*2,0 + defectiveCoords[i][0]*2))
                if coordPatch[0]>=self.imgSize//2 and coordPatch[1]<self.imgSize//2: 
                    coordInterestPatchs.append((self.imgSize+ defectiveCoords[i][1]*2,0 + defectiveCoords[i][0]*2)) 
                if coordPatch[0]<self.imgSize//2 and coordPatch[1]>=self.imgSize//2: 
                    coordInterestPatchs.append((0+ defectiveCoords[i][1]*2,self.imgSize + defectiveCoords[i][0]*2))
                if coordPatch[0]>=self.imgSize//2 and coordPatch[1]>=self.imgSize//2:
                    coordInterestPatchs.append((self.imgSize+ defectiveCoords[i][1]*2,self.imgSize + defectiveCoords[i][0]*2))
        return coordInterestPatchs    
 
         
    """
    Function for precise decision-making.
    It is based on the average scores of previous patches (image without defects).

    Args:
        img_scores: list of scores of patches
        nbPatch: number of patches
        
    Returns:
        indexDefect: list of indexes of patches considered defective
    """
    def validateAndLocateDefect(self,scores,img_scores,nbPatch):
        listScoresPatch=[[] for _ in range(nbPatch)]
        listImgScoresPatch=[[] for _ in range(nbPatch)]
        for i in range(len(img_scores)):
            listImgScoresPatch[i//(self.nbPastImages+1)].append(img_scores[i])
            listScoresPatch[i//(self.nbPastImages+1)].append(scores[i])
        indexDefect=[]
        scoresDefect=[]
        imgScoresDefects=[]
        for k,ScoresImgPath in enumerate(listImgScoresPatch):
            ScoreMean=0
            for i in range(len(ScoresImgPath)-1):
                ScoreMean=ScoreMean+ScoresImgPath[i]/(len(ScoresImgPath)-1)
            if ScoresImgPath[len(ScoresImgPath)-1]>(ScoreMean+ScoreMean*self.threshold):
                print("DEFECT CONFIRMED PATCH : "+str(k))
                indexDefect.append(k)
                imgScoresDefects.append(ScoresImgPath[len(ScoresImgPath)-1])
                scoresDefect.append(listScoresPatch[k][len(ScoresImgPath)-1])
        
        return indexDefect,scoresDefect,imgScoresDefects
            

    """
    Function for saving a defect crop and its mask.

    Args:
        image: patch with defect 
        scores: list of anomaly maps of patches
        img_scores: list of scores of patches
        i: index of the patch to save
        id_event: id of the event recorded (full frame)
        coord: coordinates of the patch
        
    Returns:
        None
    """
                  
    def saveImage(self,image,score,img_score,id_event,coord=[0,0]):
        img = np.asarray(image)
        img = np.transpose(img, (2, 0, 1))
        img = denormalization(img)
        mask = score * 1000
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask[mask > 0.8] = 1
        mask[mask <= 0.8] = 0
        
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        id_patch=insertPatchSql(self.dbHost,self.dbName, self.username, self.password, id_event, img_score,coord,precise=True)
        saveImagesDefault(image, mask, 1, id_patch,precise=True)

    """
    Function for saving the full-frame image.

    Args:
        drawing: full-frame image to be saved 
        idcamera: camera id
        
    Returns:
        id_event: id of the event recorded in the database
    """

    def saveFullFrame(self,drawing,id_event):
        now = datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")
        hour = now.strftime("%H")
        path = "data/Images/" + str(year) + "/" + str(month) + "/" + str(day) + "/" + str(
            hour) + "/"
        os.makedirs(path, exist_ok=True)
        drawing=enhanceContrastImage(drawing)
        cv2.imwrite(path +"Full_"+ str(id_event)+ "_drawing.jpg", drawing) # .png