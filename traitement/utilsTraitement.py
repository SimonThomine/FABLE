import os
import mysql.connector
import json
from datetime import datetime
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import logging
import yaml

logging.basicConfig(filename='logTest.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class App(dict):
    def __str__(self):
        return json.dumps(self)


"""
Inserts a patch event into the MySQL database.
This event corresponds to a full-frame image.

Args:
    dbHost (str): The address of the database.
    dbName (str): The name of the database.
    username (str): The username of the database.
    password (str): The password of the database.
    event_id (int): The ID of the full-frame event.
    score (float): The score of the event.
    coord (list): The list of coordinates of the patch.
        
Returns:
    inserted_id (int): The ID of the inserted patch.
"""

def insertPatchSql(dbHost,dbName,username,password,event_id, score,coord,precise=False):
    inserted_id=0
    try:
        connection = mysql.connector.connect(host=dbHost,
                                             database=dbName,
                                             user=username,
                                             password=password)
         
        mySql_insert_query = """INSERT INTO event_location (event_id,pos_x,pos_y, width,height,precise) 
                               VALUES 
                               ("""+str(event_id)+""","""+str(coord[0])+""","""+str(coord[1])+""","""+str(256)+""","""+str(256)+""","""+str(precise)+""") """
        cursor = connection.cursor()
        cursor.execute("START TRANSACTION;")
        cursor.execute(mySql_insert_query)
        cursor.execute("SELECT LAST_INSERT_ID();")
        inserted_id = cursor.fetchone()[0]
        cursor.execute("COMMIT;")
        connection.commit()
        cursor.close()

    except mysql.connector.Error as error:
        cursor.execute("ROLLBACK;")
        logging.error("Erreur lors de l'insertion du patch")
        print("Erreur lors de l'insertion de l'événement :", str(error))

    finally:
        if connection.is_connected():
            connection.close()
        return inserted_id
        
"""
Inserts a fullFrame event into the MySQL database.

Args:
    dbHost (str): The address of the database.
    dbName (str): The name of the database.
    username (str): The username of the database.
    password (str): The password of the database.
    camera (int): The ID of the camera.
        
Returns:
    inserted_id (int): The ID of the inserted event.
"""

def insertEventSql(dbHost,dbName,username,password,camera):
    inserted_id=0
    try:
        connection = mysql.connector.connect(host=dbHost,
                                             database=dbName,
                                             user=username,
                                             password=password)
         
        mySql_insert_query = """INSERT INTO event (id_camera) 
                               VALUES 
                               ("""+str(camera)+""") """

        cursor = connection.cursor()
        cursor.execute("START TRANSACTION;")
        cursor.execute(mySql_insert_query)
        cursor.execute("SELECT LAST_INSERT_ID();")
        inserted_id = cursor.fetchone()[0]
        cursor.execute("COMMIT;")
        connection.commit()
        cursor.close()

    except mysql.connector.Error as error:
        cursor.execute("ROLLBACK;")
        logging.error("Erreur lors de l'insertion de l'event")
        print("Erreur lors de l'insertion de l'événement :", str(error))

    finally:
        if connection.is_connected():
            connection.close()
        return inserted_id


"""
Saves a defect image (crop) into the MySQL database.

Args:
    image (numpy.ndarray): The image of the defect.
    mask (numpy.ndarray): The mask of the defect.
    idcamera (int): The ID of the camera.
    id_event (int): The ID of the patch.
        
Returns:
    None
"""

def saveImagesDefault(image,mask,idcamera,id_event,precise=False):
    now = datetime.now()
    year = now.strftime("%Y")
    month=now.strftime("%m")
    day=now.strftime("%d")
    hour = now.strftime("%H")
    path = "data/Images/" + str(year) + "/" + str(month) + "/" + str(day) + "/" + str(
        hour) + "/"
    os.makedirs(path, exist_ok=True)
    image=enhanceContrastImage(image)
    if (precise):
        cv2.imwrite(path + str(id_event)+"_precise.jpg", image) 
        cv2.imwrite(path + str(id_event)+"_precise_mask.jpg", mask) 
    else:
        cv2.imwrite(path + str(id_event)+".jpg", image) 
        cv2.imwrite(path + str(id_event)+"_mask.jpg", mask) 

def enhanceContrastImage(image,contrastRatio=2):
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrastRatio, beta=0)
    return adjusted_image


    
"""
Lists video files in a directory.

Args:
    directoryPath (str): The path of the directory.
        
Returns:
    videoFiles (list): The list of paths of video files.
    fileNames (list): The list of names of video files.
"""

def listFilesDirectory(directoryPath):
    onlyfiles = [f for f in listdir(directoryPath) if isfile(join(directoryPath, f))]
    videoFiles=[]
    fileNames=[]
    for i in range(len(onlyfiles)):
        _,extension=os.path.splitext(onlyfiles[i])
        if (extension=='.mp4' or extension=='.avi'):
            videoFiles.append(directoryPath+onlyfiles[i])
            fileNames.append(onlyfiles[i])
    return videoFiles,fileNames


"""
Function to crop an image vertically, taking the center of the image and 
removing 1/4 of the height from the top and bottom.

Args:
    frame (numpy.ndarray): The image to crop.
        
Returns:
    crop_img (numpy.ndarray): The cropped image.
"""

def halfHeightCrop(frame):
    y=frame.shape[0]
    x=frame.shape[1]
    crop_img = frame[int(y/4):int(y-y/4), 0:x]
    return crop_img
   
"""
Function to create a list of patches with adjustable overlap.

Args:
    frame (numpy.ndarray): The image to divide into patches.
    listImage (list): The list of patches as torch tensors.
    listCrop (list): The list of patches as numpy arrays.
    listCoord (list): The list of coordinates of patches.
    imgSize (int): The size of the patches.
    overlap (int): The overlap of patches.
    fast (bool): Whether to use fast overlapping (without overlap).
    
Returns:
    patchNbrWidth (int): The number of patches in width.
    patchNbrHeight (int): The number of patches in height.
"""

def createListPatchRecouvrement(frame,listImage,listCrop,listCoord,imgSize=256,overlap=0,fast=False):
    if (fast):
        height, width, _ = frame.shape
        patchNbrWidth = width // imgSize
        patchNbrHeight = height // imgSize
        startW = (width - imgSize * patchNbrWidth)//2
        startH = (height - imgSize * patchNbrHeight)//2
        for i in range(patchNbrWidth):
            for j in range(patchNbrHeight):                
                crop_img = frame[startH + imgSize * j:startH + imgSize * j + imgSize,
                            startW + imgSize * i:startW + imgSize * i + imgSize]
                listCoord.append([startW + imgSize * i,startH+ imgSize * j])
                listCrop.append(crop_img)
                listImage.append(torch.from_numpy(crop_img))          
    else:
        height, width = frame.shape[:2]
        patch_width, patch_height = imgSize,imgSize
        forced_overlapX=1+frame.shape[1]%patch_width//(frame.shape[1]//patch_width)
        forced_overlapY=1+frame.shape[0]%patch_height//(frame.shape[0]//patch_height)
        if (forced_overlapX<overlap):
            forced_overlapX=overlap
        if (forced_overlapY<overlap):
            forced_overlapY=overlap
        stride_x = patch_width - forced_overlapX
        stride_y = patch_height - forced_overlapY
        patchNbrHeight=0
        patchNbrWidth=0
        
        for x in range(0, width-stride_x, stride_x): # ? Je suis pas sur à 100% pour le -stride_x #
            for y in range(0, height-stride_y, stride_y): #
                x_end = min(x + patch_width, width)
                y_end = min(y + patch_height, height)
                patch = frame[y_end-patch_height:y_end, x_end-patch_width:x_end]
                listCrop.append(patch)
                listImage.append(torch.from_numpy(patch))
                listCoord.append([x_end-patch_width, y_end-patch_height])
                if x==0:
                    patchNbrHeight=patchNbrHeight+1
            patchNbrWidth=patchNbrWidth+1
    return patchNbrWidth,patchNbrHeight



"""
Function to extract patches of interest (detected defects) in an image as well as
in previous images for comparison.
We rely on the big detection and the resulting mask to extract patches.
The patches are centered on the defect and are performed on the high-resolution image.

Args:
    highResFrame (numpy.ndarray): The high-resolution image.
    coordInterestPatchs (list): The list of coordinates of patches of interest.
    listFrameHighRes (list): The list of previous high-resolution images.
    imgSize (int): The size of the patches.
        
Returns:
    batch (list): The list of patches as torch tensors.
    cropsInterestPatchs (list): The list of patches as numpy arrays.
"""


def extractInterestPatches(highResFrame,coordInterestPatchs,listFrameHighRes,imgSize=256):
    batch=[]
    cropsInterestPatchs=[]
    for coord in coordInterestPatchs:
        for frame in listFrameHighRes:
            crop_img = frame[coord[0]:coord[0] + imgSize,
                        coord[1]:coord[1] + imgSize]
            batch.append(torch.from_numpy(crop_img))
        crop_img = highResFrame[coord[0]:coord[0] + imgSize,
                        coord[1]:coord[1] + imgSize]
        batch.append(torch.from_numpy(crop_img))
        cropsInterestPatchs.append(crop_img)
    return batch,cropsInterestPatchs

        
"""
Function to normalize a batch of crops.
Works less well than individual crop normalization but very fast.

Args:
    batch (list): The list of patches as torch tensors.
    
Returns:
    batch (list): The list of normalized patches as torch tensors.
"""

def normalizeBatch(batch):
    batch = batch.permute(0,3, 1, 2).float()
    batch = (batch - 127.5) * 0.0078125
    return batch


"""
Function to crop the image to avoid detection issues on the edges.
Limits the processing area but solves the padding problem.

Args:
    anomap (numpy.ndarray): The anomaly map.
    img_size (int): The size of the image.
    crop_size (int): The size of the crop.
    
Returns:
    anomap (numpy.ndarray): The anomaly map, same size as the input but with 0s on the edges (not processed).
"""

def cropAnoMap(anomap,img_size=256,crop_size=224):
    start_row = (img_size - crop_size) // 2  
    end_row = start_row + crop_size    
    start_col = (img_size - crop_size) // 2  
    end_col = start_col + crop_size   
    anomapSmall = anomap[:,start_row:end_row, start_col:end_col]
    new_image = np.zeros(anomap.shape, dtype=np.float32)
    offset = (img_size-crop_size)//2
    new_image[:,offset:offset+crop_size, offset:offset+crop_size] = anomapSmall
    return new_image


"""
Function to read the YAML config file, safe load keeps the types.

Args:
    configFileName (str): The name of the config file.
    
Returns:
    data (dict): The YAML dictionary of the config.
"""

def readYamlConfig(configFileName):
    with open(configFileName) as f:
        data=yaml.safe_load(f)
        return data

"""
Function to read or write a threshold in the YAML file.

Args:
    threshFileName (str): The name of the threshold file.
    
Returns:
    None
"""

def writeNewThresholdYaml(threshFileName,seuil,obj):
    with open(threshFileName) as f:
        data=yaml.safe_load(f)
    data[str(obj)] = seuil
    with open(threshFileName, 'w') as f:
        yaml.dump(data,f)
        
                
"""
Function to extract the top n largest values from the heat map.

Args:
    scores (list of lists): the heat maps
    n (int): the number of largest values we want
    
Returns:
    N_img_scores (list of lists): the lists of the top n largest values
"""

def extractNMaxValues(scores,n=200):
    N_img_scores=[]
    for i,score in enumerate(scores):
        flattened_score = score.flatten()
        sorted_indices = np.argsort(flattened_score)[::-1] 
        top_n_indices = sorted_indices[:n]
        top_n_values = flattened_score[top_n_indices]
        N_img_scores.append(top_n_values)
    return N_img_scores



"""
Function to retrieve the training status of the specific model.

Args:
    dbHost (str): The address of the database.
    dbName (str): The name of the database.
    username (str): The username of the database.
    password (str): The password of the database.
    idCam (int): The ID of the camera.
    
Returns:
    trainingStatus (str): The training status ("train" or "test")
"""

def getTrainingStatus(dbHost,dbName,username,password,idCam):
    try:
        connection = mysql.connector.connect(host=dbHost,
                                             database=dbName,
                                             user=username,
                                             password=password)
         
        mySql_insert_query = """SELECT status FROM training_status WHERE id_camera="""+str(idCam)+""";"""
                               

        cursor = connection.cursor()
        cursor.execute(mySql_insert_query)
        
        trainingStatus = cursor.fetchone()[0]
        cursor.execute("COMMIT;")
        connection.commit()
        cursor.close()

    except mysql.connector.Error as error:
        cursor.execute("ROLLBACK;")
        logging.error("Erreur lors de l'insertion du patch")
        print("Erreur lors de l'insertion de l'événement :", str(error))

    finally:
        if connection.is_connected():
            connection.close()
        return trainingStatus
    

"""
Function to change the training status of the specific model.

Args:
    dbHost (str): The address of the database.
    dbName (str): The name of the database.
    username (str): The username of the database.
    password (str): The password of the database.
    idCam (int): The ID of the camera.
    trainingStatus (str): The training status ("train" or "test")
    
Returns:
    None
"""

def setTrainingStatus(dbHost,dbName,username,password,idCam,trainingStatus):
    try:
        connection = mysql.connector.connect(host=dbHost,
                                             database=dbName,
                                             user=username,
                                             password=password)
         
        mySql_insert_query = """UPDATE training_status SET status='"""+str(trainingStatus)+"""' WHERE id_camera="""+str(idCam)+""";"""
                               
        print(mySql_insert_query)
        cursor = connection.cursor()
        cursor.execute(mySql_insert_query)
        cursor.execute("COMMIT;")
        connection.commit()
        cursor.close()

    except mysql.connector.Error as error:
        cursor.execute("ROLLBACK;")
        logging.error("Erreur lors de l'insertion du patch")
        print("Erreur lors de l'insertion de l'événement :", str(error))

    finally:
        if connection.is_connected():
            connection.close()