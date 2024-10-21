'''  
    Face 
    Verification

'''

from deepface import DeepFace
import os
import cv2
from preproccess import *

# folder path
dir_path = 'C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1'
faceImage_count = 0
# Iterate through directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        faceImage_count += 1
        image_path=f"{dir_path}\{path}"
        image = cv2.imread(image_path)
        #check out if there is a face in the picture or not.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        face_inImage=0
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  #-->Draw rectangle around face
            # face verification proccess:
            result = DeepFace.verify(
                f"{dir_path}\\1.jpg",
                f"{dir_path}\\{path}", 
                model_name="Facenet", 
                detector_backend="opencv", 
                distance_metric="cosine", 
                enforce_detection=True, 
                align=True, 
                normalization="Facenet"
            )
            #if the picture verified the person successfully
            if result["verified"]:
                #write percision and result
                cv2.putText(image, f'{(1-result["distance"])*100:.2f}{result["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 1)
                #save file
                cv2.imwrite(f'face_{faceImage_count}_{face_inImage}inference.jpg', image)
                anchor = f"{dir_path}\\{path}"
            #Not successful in first attempt:
            else:
                result_anch=DeepFace.verify(
                    anchor,
                    f"{dir_path}\\{path}", 
                    model_name="Facenet", 
                    detector_backend="opencv", 
                    distance_metric="cosine", 
                    enforce_detection=True, 
                    align=True, 
                    normalization="Facenet"
                )
                #Last attempt
                if result_anch["verified"]:
                    #write percision and result
                    cv2.putText(image, f'{(1-result_anch["distance"])*100:.2f}{result_anch["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 1)
                    #save file
                    cv2.imwrite(f'face_{faceImage_count}_{face_inImage}inference.jpg', image)
                #distance is sometimes larger than 1 (cosDist = 1-cosSimilarity):
                elif result_anch["distance"]>1:
                    cv2.putText(image, f'{(result["distance"]-1)*100:.2f}{result["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 1)
                    cv2.imwrite(f'face_{faceImage_count}_{face_inImage}inference.jpg', image)
                else:
                    cv2.putText(image, f'{(1-result_anch["distance"])*100:.2f}{result_anch["verified"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 1)
                    cv2.imwrite(f'face_{faceImage_count}_{face_inImage}inference.jpg', image)

            face_inImage += 1
        
