import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
relative_path = os.path.join(current_dir, '../')
sys.path.append(relative_path)

from utilsTraitement import readYamlConfig,writeNewThresholdYaml
from utilsTraitement import listFilesDirectory
from models.trainDistillation import AnomalyDistillation

from datasetCreationModif import datasetCreator

"""
Function for creating a dataset from a directory of videos.
Calls the datasetCreator class.

Args:
    Kd: anomaly detection model
    config_file: configuration file
    reduced: boolean to indicate whether to use the reduced model or not
    
Returns:
    None
"""
def createDatasetFromDirectory(Kd,config_file,reduced=True,img_size=256):
    datasetPath=Kd.data_path+"/datasets"
    
    if (not os.path.isdir(datasetPath+"/"+str(Kd.obj)+'_'+str(img_size))):
        print("CREATING DATASET : "+Kd.data_path+"/"+str(Kd.obj)+'_'+str(img_size))
        nbImagesRequired=1000
        data = readYamlConfig(config_file)
        processType = data['Local']['processType']            
        dataCreator=datasetCreator(data,datasetPath,obj=Kd.obj,nbImagesRequired=nbImagesRequired,reduced=reduced)
        
        if (processType == 'directory'):
            directory = data['Local']['video_path']
            videoFiles,_=listFilesDirectory(directory)
                
            i=0
            while (dataCreator.compteurImagesTraining<nbImagesRequired):
                dataCreator.createDatasetForTrainingFromVideo(video_path=videoFiles[i])
                i=i+1
                if (i>len(videoFiles)-1):
                    i=0
                    print("Not Enough videos, iterating over the same videos")
        if (processType=='video'):
            video_path = data['Local']['video_path']
            dataCreator.createDatasetForTrainingFromVideo(video_path=video_path)
    
    else :
        print("EXISTING DATASET : "+Kd.data_path+"/"+str(Kd.obj)+'_'+str(img_size))


def createDatasetFromLive(Kd,config_file,reduced=True,img_size=256):
    datasetPath=Kd.data_path+"/datasets"
    
    if (not os.path.isdir(datasetPath+"/"+str(Kd.obj)+'_'+str(img_size))):
        print("CREATING DATASET : "+Kd.data_path+"/"+str(Kd.obj)+'_'+str(img_size))
        nbImagesRequired=1000
        data = readYamlConfig(config_file)
        dataCreator=datasetCreator(data,datasetPath,obj=Kd.obj,nbImagesRequired=nbImagesRequired,reduced=reduced)
        
        dataCreator.createDatasetForTrainingFromLive(data=data)
    
    else :
        print("EXISTING DATASET : "+Kd.data_path+"/"+str(Kd.obj)+'_'+str(img_size))


if __name__ == '__main__':
    print("============= TRAINING PHASE ============= ")
    
    config_file = "config.yaml"
    thresholds_file = "thresholds.yaml"
    
    data = readYamlConfig(config_file) 
    
    KdBig=AnomalyDistillation(data,inference=False,big=True,modelName=data['Models']['bigModelName'],out_indices=[])
    
    KdSmall=AnomalyDistillation(data,inference=False,big=False,modelName=data['Models']['smallModelName'],out_indices=[])
    
    img_size=data['Config']['imageSize']
    trainingMode=data['Training']['trainingMode']
    if KdBig.phase == 'train':
        if (trainingMode=="directory"):
            print("directory training Mode")
            createDatasetFromDirectory(KdBig,config_file,reduced=True,img_size=img_size)
            createDatasetFromDirectory(KdSmall,config_file,reduced=False,img_size=img_size)
        elif (trainingMode=="direct"):
            print("direct training Mode")
            createDatasetFromLive(KdBig,config_file,reduced=True,img_size=img_size)
            createDatasetFromLive(KdSmall,config_file,reduced=False,img_size=img_size)
        else: 
            print("ERROR : trainingMode not recognized")  
        KdBig.train()
        KdBig.test()
        KdSmall.train()
        KdSmall.test()    
    else :
        KdBig.test()
        KdSmall.test()
        
    writeNewThresholdYaml(thresholds_file, KdBig.threshold, KdBig.obj)
    writeNewThresholdYaml(thresholds_file, KdSmall.threshold, KdSmall.obj)
    
    







