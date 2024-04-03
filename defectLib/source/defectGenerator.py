import torch
import numpy as np
import cv2
import imgaug.augmenters as iaa
import random
import torchvision.transforms as T
import glob
from defectLib.source.perlin import rand_perlin_2d_np
import matplotlib.pyplot as plt
from defectLib.source.nsa import backGroundMask,patch_ex
from defectLib.source.cutpaste import CutPaste

class TexturalAnomalyGenerator():
    def __init__(self, resize_shape=None,dtd_path="../../datasets/dtd/images"):
        
        
        self.resize_shape=resize_shape
        self.anomaly_source_paths = sorted(glob.glob(dtd_path+"/*/*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-10,10),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      ]
        
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    def getDtdImage(self):
        randIndex=random.randint(0, len(self.anomaly_source_paths)-1)
        image=cv2.imread(self.anomaly_source_paths[randIndex])
        image=cv2.resize(image, dsize=(self.resize_shape[0], self.resize_shape[1]))
        aug=self.randAugmenter()
        image=aug(image=image)
        return image 
    

class StructuralAnomalyGenerator():
    def __init__(self,resize_shape=None):
        
        self.resize_shape=resize_shape
        self.augmenters = [iaa.Fliplr(0.5),  
                            iaa.Affine(rotate=(-45, 45)),  
                            iaa.Multiply((0.8, 1.2)),  
                            iaa.MultiplySaturation((0.5, 1.5)), 
                            iaa.MultiplyHue((0.5, 1.5))
                      ]
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def generateStructuralDefect(self,image):
        aug=self.randAugmenter()
        image_array=(image.permute(1,2,0).numpy()*255).astype(np.uint8)# # *


        image_array=aug(image=image_array)

        height, width, _ = image_array.shape
        grid_size = 8
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        grid = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = image_array[i * cell_height: (i + 1) * cell_height,
                                j * cell_width: (j + 1) * cell_width, :]
                grid.append(cell)
        
        np.random.shuffle(grid)
        
        reconstructed_image = np.zeros_like(image_array)
        for i in range(grid_size):
            for j in range(grid_size):
                reconstructed_image[i * cell_height: (i + 1) * cell_height,
                                    j * cell_width: (j + 1) * cell_width, :] = grid[i * grid_size + j]
        return reconstructed_image



class DefectGenerator():

    def __init__(self, resize_shape=None,dtd_path="../../datasets/dtd/images"): 


        self.texturalAnomalyGenerator=TexturalAnomalyGenerator(resize_shape,dtd_path)
        self.structuralAnomalyGenerator=StructuralAnomalyGenerator(resize_shape)
        self.cutpaste=CutPaste()
        
        self.resize_shape=resize_shape
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.toTensor=T.ToTensor()
        
    def generateMask(self,bMask):
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        msk = (perlin_thr).astype(np.float32) 
        msk=torch.from_numpy(msk).permute(2,0,1)
        if (len(bMask)>0):
            msk=bMask*msk
        return msk
    
    def generateTexturalDefect(self, image,bMask=[]):
        msk=torch.zeros((self.resize_shape[0], self.resize_shape[1]))
        while (torch.count_nonzero(msk)<100):
            msk=self.generateMask(bMask)*255.0
        texturalImg=self.texturalAnomalyGenerator.getDtdImage()
        texturalImg=torch.from_numpy(texturalImg).permute(2,0,1)/255.0
        mskDtd=texturalImg*(msk)
        
        image = image * (1 - msk)+  (mskDtd) 
        return image ,msk
    
    def generateStructuralDefect(self, image,bMask=[]):
        msk=torch.zeros((self.resize_shape[0], self.resize_shape[1]))
        while (torch.count_nonzero(msk)<100):
            msk=self.generateMask(bMask)*255.0
        structuralImg=self.structuralAnomalyGenerator.generateStructuralDefect(image)/255.0
        structuralImg=torch.from_numpy(structuralImg).permute(2,0,1)
        
        mskDtd=structuralImg*(msk)
        image = image * (1 - msk)+  (mskDtd) 
        return image ,msk
    
    
    def generateBlurredDefectiveImage(self, image,bMask=[]):
        msk=torch.zeros((self.resize_shape[0], self.resize_shape[1]))
        while (torch.count_nonzero(msk)<100):
            msk=self.generateMask(bMask)*255.0
        randGaussianValue = random.randint(0, 5)*2+21 
        transform = T.GaussianBlur(kernel_size=(randGaussianValue, randGaussianValue), sigma=11.0)
        imageBlurred = transform(image)
        imageBlurred=imageBlurred*(msk)
        image=image*(1-msk)

        image=image+imageBlurred

        return image,msk

    def generateNsaDefect(self, image,bMask):
        image = np.expand_dims(np.array(image),2) if len(np.array(image).shape)==2 else np.array(image)
        image,msk=patch_ex(image,backgroundMask=bMask)
        transform=T.ToTensor()
        image = transform(image)
        msk = transform(msk)*255.0
        return image,msk
    
    def generateCutPasteDefect(self, image,bMask):
        msk=np.zeros((self.resize_shape[0], self.resize_shape[1]))
        while (np.count_nonzero(msk)<100):
            defect,cpmsk=self.cutpaste.cutpaste(image)
            msk=bMask*np.expand_dims(np.array(cpmsk),axis=2)
        image=np.array(defect)*bMask + np.array(image)*(1-bMask)
        transform=T.ToTensor()
        image = transform(image)
        msk = transform(msk)
        return image,msk
    
    
    def genSingleDefect(self,image,label,mskbg):
        if label.lower() not in ["textural","structural","blurred","nsa","cutpaste"]:
            raise ValueError("The defect type should be in ['textural','structural','blurred','nsa','cutpaste']")
         
        if (label.lower()=="textural" or label.lower()=="structural" or label.lower()=="blurred"):
            imageT=self.toTensor(image)
            bmask=self.toTensor(mskbg)
            if (label.lower()=="textural"):
                return self.generateTexturalDefect(imageT,bmask)
            elif (label.lower()=="structural"):
                return self.generateStructuralDefect(imageT,bmask)
            elif (label.lower()=="blurred"):
                return self.generateBlurredDefectiveImage(imageT,bmask)
        elif (label.lower()=="nsa"):
            return self.generateNsaDefect(image,mskbg) 
        elif (label.lower()=="cutpaste"):
            return self.generateCutPasteDefect(image,mskbg)


    def genDefect(self,image,defectType,category="",return_list=False):
        mskbg=backGroundMask(image,obj=category) 
        if not return_list:
            if (len(defectType)>1):
                index=np.random.randint(0,len(defectType))
                label=defectType[index]
            else:
                label=defectType[0]
            return self.genSingleDefect(image,label,mskbg)            
        if return_list: 
            defectImages=[]
            defectMasks=[]
            for label in defectType:
                defectImage,defectMask=self.genSingleDefect(image,label,mskbg)
                defectImages.append(defectImage)
                defectMasks.append(defectMask)
            return defectImages,defectMasks
