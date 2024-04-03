import sys
sys.path.append('../')

from utilsTraitement import listFilesDirectory
from datasetCreationModif import datasetCreator
from utilsTraitement import readYamlConfig

if __name__ == '__main__':
    directory = "data/DG_IDS_videos/"
    videoFiles=listFilesDirectory(directory)

    config_file = "config.yaml"
    data = readYamlConfig(config_file)
    size=data['Config']['imageSize']
    
    dataCreatorBig=datasetCreator(data,"data/datasets",obj="DG_ids",nbImagesRequired=200,reduced=False)
    print("CREATING DATASET : "+"data/datasets/DG_ids_"+str(size))
    for video in videoFiles[0]:
        dataCreatorBig.createDatasetDgDatasetFromVideo(video)
    dataCreatorSmall=datasetCreator(data,"data/datasets",obj="DG_ids_big",nbImagesRequired=200)
    print("CREATING DATASET : "+"data/datasets/DG_ids_big_"+str(size))
    for video in videoFiles[0]:
        dataCreatorSmall.createDatasetDgDatasetFromVideo(video)
