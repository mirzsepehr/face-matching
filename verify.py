import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import BatchNormalization
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from preproccess import *
from deepface import DeepFace

def resize_image(img, target_size):
    """
    Resizes an image to 
    the specified target size.
    """
    resizedImage=cv2.resize(img, target_size)
    return resizedImage


def normalize_brightness(img):
    # Convert image to float32 for calculations
    img = img.astype('float32')

    # Calculate mean and standard deviation
    mean, std = cv2.meanStdDev(img)

    # Normalize the image
    img = (img - mean) / std

    # Scale the image back to 0-255 range
    # img = np.clip(img * 127.5 + 127.5, 0, 255).astype('uint8')

    return img

def increase_contrast(image, alpha=2, beta=0):
    """Increases the contrast of an image.

    Args:
        image: The input image.
        alpha: Contrast control (1.0 for no change, >1.0 to increase contrast).
        beta: Brightness control (0 for no change, >0 to increase brightness).

    Returns:
        The contrast-adjusted image.
    """

    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image



'''
Section 1:    
    Preprocessing images
'''
for num in range (0,23):
    imagePath = f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\5\\{num+1}.jpg"
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # normalized_brightness_image = normalize_brightness(gray)
    # contrast_adjusted_image = increase_contrast(normalized_brightness_image)


    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")      # --> implementation of haar cascade method!
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    ) 

    print("Found {0} Faces!".format(len(faces)))


    # cnt=0
    # for (x, y, w, h) in faces:                                                                               # --> save the extracted faces
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     roi_color = image[y:y + h, x:x + w] 
    #     resized_image=resize_image(roi_color, (400, 400))                                                    # --> make all images the same size
    #     # print("Saving locally!!!")
    #     cv2.imwrite(f'face{num}_{cnt}.jpg', resized_image) 
    #     cnt+=1

# result_anch=DeepFace.verify(
#     "C:\\Users\\Sepehr\\Desktop\\project1\\Images\\5\\1.jpg",
#     "C:\\Users\\Sepehr\\Desktop\\project1\\Images\\5\\2.JPG", 
#     model_name="Facenet", 
#     detector_backend="opencv", 
#     distance_metric="cosine", 
#     enforce_detection=True, 
#     align=True, 
#     normalization="Facenet"
# )

# print(result_anch["distance"])




'''
Section 2:    
    Image Verification

'''

# from deepface import DeepFace

# for num in range (0,5):
#     imagePath = f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\{num+1}.jpg"
#     image = cv2.imread(imagePath)


#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                                           # --> convert image to the gray scale

#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")      # --> implementation of haar cascade method!
#     faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.3,
#             minNeighbors=3,
#             minSize=(30, 30)
#     ) 

#     print("Found {0} Faces!".format(len(faces)))




#     cnt=0
#     for (x, y, w, h) in faces:                                                                               # --> save the extracted faces
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         result_anch = DeepFace.verify("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\1.jpg",
#                          f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\{num+1}.jpg", 
#                          model_name="Facenet", 
#                          detector_backend="opencv", 
#                          distance_metric="cosine", 
#                          enforce_detection=True, 
#                          align=True, 
#                          normalization="Facenet")
#         if result_anch["verified"] == True:
#             cv2.putText(image, f'{(100-(int(1000*result_anch["distance"])/10))}{result_anch["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#             print("Saving verified pictures locally . . .")
#             cv2.imwrite(f'face{num}_{cnt}_pr.jpg', image)
#             anchor = f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\{num+1}.jpg" 
#         else:
#             result_anch=DeepFace.verify(
#                 anchor,
#                 f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\{num+1}.jpg", 
#                 model_name="Facenet", 
#                 detector_backend="opencv", 
#                 distance_metric="cosine", 
#                 enforce_detection=True, 
#                 align=True, 
#                 normalization="Facenet"
#             )
#             cv2.putText(image, f'{(100-(int(1000*result_anch["distance"])/10))}{result_anch["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#             cv2.imwrite(f'face{num}_{cnt}_pr.jpg', image)
#         cnt+=1
# result = DeepFace.verify(
#     "C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\3.jpg",
#     "C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\4.jpg",
#     model_name="Facenet",
#     detector_backend="opencv",
#     distance_metric="cosine",
#     enforce_detection=True, 
#     align=True, 
#     normalization="Facenet"
#     )
# print(result["verified"], result["distance"])



# _______________________________________________________________________________________________________________________

# cnt=0
# for num in range (0,3):
#     imagePath = f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\2\\{num+1}.jfif"
#     image = cv2.imread(imagePath)


#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                                           # --> convert image to the gray scale

#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")      # --> implementation of haar cascade method!
#     faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.3,
#             minNeighbors=3,
#             minSize=(30, 30)
#     ) 

#     print("Found {0} Faces!".format(len(faces)))
#     cnt=0
#     for (x, y, w, h) in faces:                                                                               # --> save the extracted faces
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         result_anch = DeepFace.verify("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\1.jpg",
#                          f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\2\\{num+1}.jfif", 
#                          model_name="Facenet", 
#                          detector_backend="opencv", 
#                          distance_metric="cosine", 
#                          enforce_detection=True, 
#                          align=True, 
#                          normalization="Facenet")
#         if result_anch["verified"] == True:
#             cv2.putText(image, f'{(100-(int(1000*result_anch["distance"])/10)):.1f}{result_anch["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#             print("Saving verified pictures locally . . .")
#             cv2.imwrite(f'face{num}_{cnt}_pr.jpg', image)
#             anchor = f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\{num+1}.jpg" 
#         else:
#             result_anch=DeepFace.verify(
#                 anchor,
#                 f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\2\\{num+1}.jfif", 
#                 model_name="Facenet", 
#                 detector_backend="opencv", 
#                 distance_metric="cosine", 
#                 enforce_detection=True, 
#                 align=True, 
#                 normalization="Facenet"
#             )
#             if result_anch["distance"] > 1:
#                 cv2.putText(image, f'{((int(1000*result_anch["distance"]))/10)-100:.1f}{result_anch["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#                 cv2.imwrite(f'face{num}_{cnt}_pr.jpg', image)
#             else:
#                 cv2.putText(image, f'{(100-(int(1000*result_anch["distance"])/10)):.1f}{result_anch["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#                 cv2.imwrite(f'face{num}_{cnt}_pr.jpg', image)
#         cnt+=1

# result_anch = DeepFace.verify("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\1.jpg",
#                     f"C:\\Users\\Sepehr\\Desktop\\project1\\Images\\2\\{1}.jfif", 
#                     model_name="Facenet", 
#                     detector_backend="opencv", 
#                     distance_metric="cosine", 
#                     enforce_detection=True, 
#                     align=True, 
#                     normalization="Facenet")

# print(f"#####{result_anch['distance']} #####")




# image2=cv2.imread("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\3.JPG")
# prep = DeepFace.preprocessing.normalize_input(image2, normalization="Facenet")
# # print(type(prep))
# img_float32 = np.float32(prep)
# image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)
# cv2.imwrite("prep.jpg", image)



    
# image = cv2.imread('C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\4.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# enhanced_image = clahe.apply(gray)
# cv2.imwrite(f'face3_illumination_NORM.jpg', enhanced_image)

# result = DeepFace.verify("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\1.jpg",
#                          "C:\\Users\\Sepehr\\Desktop\\project1\\face3_illumination_NORM.jpg", 
#                          model_name="Facenet", 
#                          detector_backend="opencv", 
#                          distance_metric="cosine", 
#                          enforce_detection=True, 
#                          align=True, 
#                          normalization="Facenet")
# print(f"illumination solution is {result['verified']}")

