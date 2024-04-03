import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from datasets.perlin import rand_perlin_2d_np
from defectLib.source.defectGenerator import DefectGenerator

class SyntheticDefectGenerator(Dataset):

    def __init__(self, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.resize_shape=resize_shape
        self.defGen=DefectGenerator(resize_shape=self.resize_shape,dtd_path=anomaly_source_path)
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

    def transform_imageFromVideo(self, image):
        
        defect,msk=self.defGen.genDefect(image,defectType=["nsa","textural"])
        defect = (defect.permute(1, 2, 0).numpy())
        msk = (msk.permute(1, 2, 0).numpy())
        
        return defect, msk, True
        
    

    
    

