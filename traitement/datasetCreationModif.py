import os
import cv2
import numpy as np
import torch
from PIL import Image

from traitement.SyntheticDefectGenerator import SyntheticDefectGenerator
from utilsTraitement import halfHeightCrop

import time
from models.resnet_reduced_backbone import modified_resnet18
from torchvision import transforms as T
from models.teacherTimm import featureExtractor

from harvesters.core import Harvester
import random


class datasetCreator(): 
    def __init__(self,data,data_path,obj,nbImagesRequired=1000,reduced=True):
           
        self.data_path=data_path
        self.obj=obj
        self.perinNoise_path=data['Training']['perinNoise_path']
        self.reduced = reduced 
        self.half=data['Config']['half'] 
        self.img_size=data['Config']['imageSize']
        self.featureExtractorTrain=data['Training']['featureExtractorTrain']
        self.nbImagesRequired=nbImagesRequired
        self.compteurImagesTraining=0
        self.listCrop=[]
        self.path=data_path+"/"+obj+"_"+str(self.img_size)
         
        self.nbCropsRequired=50
        
    """
    Function for creating the training dataset from a video.
    This function is called multiple times to create a complete training dataset.

    Args:
        video_path: path of the video to process
        
    Returns:
        None
    """    
    def createDatasetForTrainingFromVideo(self,video_path):
        cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        nbFrame = 0
        mvtDetected=False
        previous_frame=np.zeros([1,1])
        cptImgSelect=0
        modulo=3
        isDefect=False
        perinNoiseGenerator = SyntheticDefectGenerator(self.perinNoise_path, resize_shape=[self.img_size, self.img_size])
        
        while (cap.isOpened()):
            ret, frame = cap.read()
            if(ret == False):
                cap.release()
            if (self.half==True and ret):
                frame=halfHeightCrop(frame)
            if (ret and self.reduced == True):
                frame=cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))  
            if  ret : 
                mvtDetected=basicMotionDetection(frame,previous_frame)
            previous_frame=frame

            if ret == True and nbFrame > 10 and nbFrame%5==0 and mvtDetected: # on prend 1 frame sur 5 pour un max de variété
                         
                
                listCropImg=self.getRandomOffSetCrops(frame,offSetMaxSize=30)
                for i,crop_img in enumerate(listCropImg):
                    if (i%5!=4):
                        self.listCrop.append(crop_img)
                    else :
                        if (not isDefect):
                            pathAct = self.path + "/test/good/"
                            os.makedirs(pathAct, exist_ok=True)
                            cv2.imwrite(pathAct + str(self.compteurImagesTraining) + ".png", crop_img)
                            isDefect=True
                        else:
                            self.handleTestImage(perinNoiseGenerator,crop_img)
                            isDefect=False
                    self.compteurImagesTraining = self.compteurImagesTraining+1
                        
                cptImgSelect=cptImgSelect+1
                if (cptImgSelect>=modulo):
                    cptImgSelect=0 
                if self.compteurImagesTraining > self.nbImagesRequired:
                    if (self.featureExtractorTrain):
                        print("IMAGES GATHERED ==> EXTRACT MOST DIVERSE")
                        extractFeat=featureExtractor(modelName="Resnet18",listImages=self.listCrop,nbCrops=self.nbCropsRequired)
                        listDiverseCrop=extractFeat.selectMostDiverseImages()
                        listDiverseCrop = random.sample(listDiverseCrop, self.nbCropsRequired)
                    else :
                        print("IMAGES GATHERED ==> EXTRACT RANDOM")
                        listDiverseCrop = random.sample(self.listCrop, self.nbCropsRequired)
                    compteur=0
                    for crop in listDiverseCrop:
                        pathAct = self.path + "/train/good/"
                        os.makedirs(pathAct, exist_ok=True)
                        cv2.imwrite(pathAct + str(compteur) + ".png", crop)
                        compteur=compteur+1
                    cap.release()
            nbFrame = nbFrame + 1

    """
    Function for creating the training dataset from live GenICam stream.

    Args:
        data: the dictionary from the JSON config file
        
    Returns:
        None
    """
 
    def createDatasetForTrainingFromLive(self,data):
        h = Harvester()
        h.add_file(data['Genicam']['cti'])
        h.update()
        ia = h.create()
        ia.remote_device.node_map.PixelFormat.value = 'BayerRG8'
        ia.remote_device.node_map.ExposureTime.set_value(int(data['Genicam']['exposure'])) 
        ia.start()
        compteurFrame=0
        previous_frame = np.zeros([1, 1])
        
        nbFrame = 0
        mvtDetected=False
        previous_frame=np.zeros([1,1])
        cptImgSelect=0
        modulo=3
        isDefect=False
        perinNoiseGenerator = createPerinAugmentedDataset(self.perinNoise_path, resize_shape=[self.img_size, self.img_size])
        while (True):
            compteurFrame=compteurFrame+1
            with ia.fetch() as buffer:
                component = buffer.payload.components[0]
                _2d = component.data.reshape(int(data['Genicam']['height']), int(data['Genicam']['width']))
                frame = _2d
                if 'Bayer' in ia.remote_device.node_map.PixelFormat.value:
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)  
            if (self.half==True):
                frame=halfHeightCrop(frame)
            if (self.reduced == True):
                frame=cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))  
            mvtDetected=basicMotionDetection(frame,previous_frame)
            previous_frame=frame

            if nbFrame > 10 and nbFrame%5==0 and mvtDetected: 
                         
                
                listCropImg=self.getRandomOffSetCrops(frame,offSetMaxSize=30)
                for i,crop_img in enumerate(listCropImg):
                    if (i%5!=4):
                        self.listCrop.append(crop_img)
                    else :
                        if (not isDefect):
                            pathAct = self.path + "/test/good/"
                            os.makedirs(pathAct, exist_ok=True)
                            cv2.imwrite(pathAct + str(self.compteurImagesTraining) + ".png", crop_img)
                            isDefect=True
                        else:
                            self.handleTestImage(perinNoiseGenerator,crop_img) 
                            isDefect=False
                    self.compteurImagesTraining = self.compteurImagesTraining+1
                        
                cptImgSelect=cptImgSelect+1
                if (cptImgSelect>=modulo):
                    cptImgSelect=0 
                if self.compteurImagesTraining > self.nbImagesRequired:
                    if (self.featureExtractorTrain):
                        print("IMAGES GATHERED ==> EXTRACT MOST DIVERSE")
                        extractFeat=featureExtractor(modelName="Resnet18",listImages=self.listCrop,nbCrops=self.nbCropsRequired)
                        listDiverseCrop=extractFeat.selectMostDiverseImages()
                        listDiverseCrop = random.sample(listDiverseCrop, self.nbCropsRequired)
                    else :
                        print("IMAGES GATHERED ==> EXTRACT RANDOM")
                        listDiverseCrop = random.sample(self.listCrop, self.nbCropsRequired)
                    compteur=0
                    for crop in listDiverseCrop:
                        pathAct = self.path + "/train/good/"
                        os.makedirs(pathAct, exist_ok=True)
                        cv2.imwrite(pathAct + str(compteur) + ".png", crop)
                        compteur=compteur+1
                    ia.stop()
                    ia.destroy()
                    h.reset()
                    break
                    
            nbFrame = nbFrame + 1


    """
    Function for creating the domain generalization training dataset from a video.
    This function is called multiple times to create a complete training dataset.

    Args:
        video_path: path of the video to process
        
    Returns:
        None
    """

    def createDatasetDgDatasetFromVideo(self,video_path):
        listCrop=[]
        compteurImagesAct=0
        cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        nbFrame = 0
        mvtDetected=False
        previous_frame=np.zeros([1,1])
        cptImgSelect=0
        modulo=3
        alreadySaved=False
        perinNoiseGenerator = createPerinAugmentedDataset(self.perinNoise_path, resize_shape=[self.img_size, self.img_size])
        while (cap.isOpened()):
            ret, frame = cap.read()
            if(ret == False):
                cap.release()
            if (self.half and ret):
                frame=halfHeightCrop(frame)
            if  ret : 
                mvtDetected=basicMotionDetection(frame,previous_frame)
            if (ret and self.reduced == True):
                frame=cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))  
            previous_frame=frame
            if ret == True and nbFrame > 10 and nbFrame%5==0 and mvtDetected: 
                
                height, width, _ = frame.shape

                patchNbrWidth = width // self.img_size
                patchNbrHeight = height // self.img_size
                diffWidth = width - self.img_size * patchNbrWidth
                diffHeight = height - self.img_size * patchNbrHeight
                
                for i in range(patchNbrWidth):
                    for j in range(patchNbrHeight):
                        crop_img = frame[diffHeight // 2 + self.img_size * j:diffHeight // 2 + self.img_size * j + self.img_size,
                                    diffWidth // 2 + self.img_size * i:diffWidth // 2 + self.img_size * i + self.img_size]
                        if ((i+j)%modulo!=cptImgSelect):
                            continue  
                        if (compteurImagesAct%5!=4):
                            listCrop.append(crop_img)  
                                                
                        if (compteurImagesAct%5==4):
                            if (compteurImagesAct % 2 == 0):
                                pathAct = self.path + "/test/good/"
                                os.makedirs(pathAct, exist_ok=True)
                                cv2.imwrite(pathAct + str(self.compteurImagesTraining) + ".png", crop_img)
                            else:
                                pathAct = self.path + "/test/anomaly/"
                                os.makedirs(pathAct, exist_ok=True)
                                pathMask = self.path + "/ground_truth/anomaly/"
                                os.makedirs(pathMask, exist_ok=True)
                                anomaly_source_idx = torch.randint(0, len(perinNoiseGenerator.anomaly_source_paths),
                                                                    (1,)).item()
                                image, augmented_image, anomaly_mask, has_anomaly = perinNoiseGenerator.transform_imageFromVideo(
                                    crop_img,
                                    perinNoiseGenerator.anomaly_source_paths[
                                        anomaly_source_idx])
                                if has_anomaly:
                                    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                                    augmented_image = Image.fromarray((augmented_image * 255).astype(np.uint8))
                                    anomaly_mask = Image.fromarray((anomaly_mask.squeeze() * 255).astype(np.uint8))
                                    augmented_image.save(pathAct + str(self.compteurImagesTraining) + ".png")
                                    anomaly_mask.save(pathMask + str(self.compteurImagesTraining) + "_mask.png")
                        compteurImagesAct = compteurImagesAct + 1
                        self.compteurImagesTraining=self.compteurImagesTraining+1
                cptImgSelect=cptImgSelect+1
                if (cptImgSelect>=modulo):
                    cptImgSelect=0 
                if compteurImagesAct > self.nbImagesRequired:
                    extractFeat=featureExtractor(modelName="Resnet18",listImages=listCrop,nbCrops=self.nbCropsRequired)
                    listDiverseCrop=extractFeat.selectMostDiverseImages()
                    for crop in listDiverseCrop:
                        pathAct = self.path + "/train/good/"
                        os.makedirs(pathAct, exist_ok=True)
                        cv2.imwrite(pathAct + str(self.compteurImagesTraining) + ".png", crop)
                        self.compteurImagesTraining=self.compteurImagesTraining+1
                        alreadySaved=True
                    cap.release()
            nbFrame = nbFrame + 1
        if(not alreadySaved and len(listCrop)> self.nbCropsRequired):
            print("IMAGES GATHERED ==> EXTRACT MOST DIVERSE")
            extractFeat=featureExtractor(modelName="Resnet18",listImages=listCrop,nbCrops=self.nbCropsRequired)
            listDiverseCrop=extractFeat.selectMostDiverseImages()
            for crop in listDiverseCrop:
                pathAct = self.path + "/train/good/"
                os.makedirs(pathAct, exist_ok=True)
                cv2.imwrite(pathAct + str(self.compteurImagesTraining) + ".png", crop)
                self.compteurImagesTraining=self.compteurImagesTraining+1
                alreadySaved=True
        if (len(listCrop)< self.nbCropsRequired*3):
            print("not enough images gathered from the video")


    """
    Function to create a list of patches from a frame by adding
    a random offset for each patch.

    Args:
        frame: frame to be cropped
        offSetMaxSize: maximum size of the offset
        
    Returns:
        listCropImg: list of patches
    """
    def getRandomOffSetCrops(self,frame,offSetMaxSize=30):
        listCropImg=[]
        height, width, _ = frame.shape
        patchNbrWidth = width // self.img_size
        patchNbrHeight = height // self.img_size
        diffWidth = width - self.img_size * patchNbrWidth
        diffHeight = height - self.img_size * patchNbrHeight
        randomOffSetX=0
        randomOffSetY=0
        for i in range(patchNbrWidth):
            for j in range(patchNbrHeight):
                if (i==patchNbrWidth-1):
                    randomOffSetX=0
                else:
                    randomOffSetX=np.random.randint(0,offSetMaxSize)
                if (j==patchNbrHeight-1):
                    randomOffSetY=0
                else: 
                    randomOffSetY=np.random.randint(0,offSetMaxSize)
                yCoord=diffHeight // 2 + self.img_size * j+randomOffSetY
                xCoord=diffWidth // 2 + self.img_size * i+randomOffSetX
                crop_img = frame[yCoord:yCoord + self.img_size,
                                xCoord:xCoord+ self.img_size]
                listCropImg.append(crop_img)
        return listCropImg


    """
    Function to handle test images, with the option to generate a custom defect.

    Args:
        perinNoiseGenerator: defect generator
        crop_img: patch to process
        
    Returns:
        None
    """
    def handleTestImage(self,perinNoiseGenerator,crop_img):
        pathAct = self.path + "/test/anomaly/"
        os.makedirs(pathAct, exist_ok=True)
        pathMask = self.path + "/ground_truth/anomaly/"
        os.makedirs(pathMask, exist_ok=True)

        augmented_image, anomaly_mask, has_anomaly = perinNoiseGenerator.transform_imageFromVideo(crop_img)
        
        if has_anomaly:
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)

            augmented_image = Image.fromarray((augmented_image * 255).astype(np.uint8))

            anomaly_mask = Image.fromarray((anomaly_mask.squeeze() * 255).astype(np.uint8))

            augmented_image.save(pathAct + str(self.compteurImagesTraining) + ".png")
            anomaly_mask.save(pathMask + str(self.compteurImagesTraining) + "_mask.png")



"""
Function for motion detection based on the difference between two successive frames.

Args:
    frame: current frame
    previous_frame: previous frame
    
Returns:
    boolean: True if motion detected, False otherwise
"""

def basicMotionDetection(frame,previous_frame):
    if (previous_frame.shape[0]==1):
        return False
    prepared_frame=cv2.resize(frame,(int(frame.shape[1]/4),int(frame.shape[0]/4)))
    prepared_frame = cv2.cvtColor(prepared_frame, cv2.COLOR_BGR2GRAY)

    previous_frame = cv2.resize(previous_frame, (int(frame.shape[1] / 4), int(frame.shape[0] / 4)))
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
    
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)
    
    thresh_frame = cv2.threshold(src=diff_frame, thresh=10, maxval=255, type=cv2.THRESH_BINARY)[1]
    
    if (np.count_nonzero(thresh_frame)>10000):
        return True
    return False


 
        
"""
The feature extractor class is used to extract image features for the purpose of 
selecting the most relevant images for model training.
""" 

class featureExtractor(): 
    def __init__(self,modelName="Resnet18",listImages=[],nbCrops=50):
        self.device='cuda'
        if (modelName=="Resnet18"):
            self.model_t = modified_resnet18().to(self.device)

        elif(modelName=="efficientnet-b0"):
            self.model_t=featureExtractor(backbone_name="efficientnet_b0",out_indices=[3,4]).to(self.device)
        else :
            print("no model found for feature extraction, random process")   
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()
        self.nbCrops=nbCrops
        self.transform_x = T.Compose([T.ToPILImage(),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.listImages=listImages
        self.listFeatures=[]
        
        self.selectedImages=[]
        
    """
    Function that extracts features and compares them for each channel.
    This function serves as the link between different functions of the class.

    Args:
        None
        
    Returns:
        selectedImages: list of selected images
    """
    def selectMostDiverseImages(self):
        self.extractAllFeatures()
        self.compareFeaturesForEveryChannel()
        return self.selectedImages 
    
    """
    Function for extracting features from all images.

    Args:
        None
        
    Returns:
        None
    """
    def extractAllFeatures(self):
        for img in self.listImages:
            features=self.extractFeaturesImage(img)
            self.listFeatures.append(features)
    """
    Function for extracting features from a single image.

    Args:
        data: image to process
        
    Returns:
        features: extracted features
    """
    def extractFeaturesImage(self,data):
        dataNorm=self.transform_x(data)
        dataNorm=dataNorm.to(self.device).unsqueeze(0)
        features=self.model_t(dataNorm)
        return features
    
    """
    Function for comparing features of all images.

    Args:
        None
        
    Returns:
        None
    """
    def compareFeaturesForEveryChannel(self):
        
        if (len(self.listFeatures))==0:
            return
        
        nbChannels=len(self.listFeatures[0])
        for i in range(nbChannels):
            listFeaturesChannel=[]
            for features in self.listFeatures:
                listFeaturesChannel.append(features[i].cpu().numpy())
            if (i==0):
                continue    
            selected_indices=self.diverse_exemplar_selection(listFeaturesChannel,self.nbCrops)

            selected_imagesChan = [self.listImages[i] for i in selected_indices]
            
            selected_indices = sorted(list(set(selected_indices)), reverse=True)
            self.listFeatures = np.delete(self.listFeatures, np.array(selected_indices), axis=0)
            self.listImages=np.delete(self.listImages,np.array(selected_indices),axis=0)
            
            for img in selected_imagesChan:
                self.selectedImages.append(img)
     
    """
    Function for calculating to select the most diverse images.

    Args:
        features: features to process
        num_examples: number of examples to select
        
    Returns:
        selected_indices: list of indices of selected images
    """

    def diverse_exemplar_selection(self,features, num_examples):
        features=np.array(features) 
        num_total_examples = features.shape[0]
        selected_indices = []

        first_index = np.random.randint(num_total_examples)
        selected_indices.append(first_index)

        similarity_matrix = np.zeros((num_total_examples, num_total_examples))
        for i in range(num_total_examples):
            for j in range(num_total_examples):
                similarity_matrix[i, j] = np.linalg.norm(features[i] - features[j])  

        for _ in range(num_examples - 1):
            non_selected_indices = [i for i in range(num_total_examples) if i not in selected_indices]

            similarities = np.mean(similarity_matrix[selected_indices][:, non_selected_indices], axis=0)

            new_index = non_selected_indices[np.argmax(similarities)]

            selected_indices.append(new_index)
        return selected_indices
                