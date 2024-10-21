import cv2
import os
import tensorflow as tf
import inception_resnet_v1
import numpy as np
import requests
from super_image import EdsrModel, ImageLoader
import torch
from cv2 import dnn_superres
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
from torchvision import transforms


def preprocessing(img):
    image_plot = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imageSize = (tf.convert_to_tensor(image_plot.shape[:-1]) // 4) * 4
    cropped_image = tf.image.crop_to_bounding_box(
        img, 0, 0, imageSize[0], imageSize[1])
    preprocessed_image = tf.cast(cropped_image, tf.float32)
    return tf.expand_dims(preprocessed_image, 0)

def grayscale_to_3channels(image):
    """
    
    Converts a grayscale PIL Image 
    to a 3-channel image.
    
    """
    
    if image.mode != 'L':
        raise ValueError("Image must be in grayscale mode (L)")

    return Image.merge("RGB", (image, image, image))

def normalize_illumination(image):
    """Normalizes the illumination of an image using PIL."""

    # Convert PIL Image to NumPy array
    img_array = np.array(image)

    # Calculate mean and standard deviation of pixel values
    mean = np.mean(img_array)
    std = np.std(img_array)

    # Normalize pixel values
    img_array = (img_array - mean) / std

    # Scale pixel values to [0, 255]
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255

    # Convert NumPy array back to PIL Image
    normalized_image = Image.fromarray(img_array.astype('uint8'))

    return normalized_image




 

'''
Detect faces
'''

def resize_image(img, target_size):
    """
    Resizes an image to 
    the specified target size.
    """
    resizedImage=cv2.resize(img, target_size, cv2.INTER_AREA)
    return resizedImage

root_dir_path = "C:\\Users\\Sepehr\\Desktop\\project1\\Images"

# my SR Model
sr = dnn_superres.DnnSuperResImpl_create()
edsrModelPath = "EDSR_x4.pb"
sr.readModel(edsrModelPath)
sr.setModel("edsr", 4)


# walk throug data path and find faces
for root, dirs, files in os.walk(root_dir_path, topdown=False):
   for name in dirs:
        current_directory = os.path.join(root, name)
        for path in os.listdir(current_directory):
            # check if current path is a file
            if os.path.isfile(os.path.join(current_directory, path)):
                # print(os.path.join(current_directory, path))
                img = cv2.imread(os.path.join(current_directory, path))
                # upsampledImg = sr.upsample(img)
                # resizedImg = cv2.resize(img, dsize=None, fx=4, fy=4)
                face_detector = cv2.FaceDetectorYN.create(
                    model=r"C:\Users\Sepehr\Desktop\project1\face_detection_yunet_2022mar.onnx",
                    config="",
                    input_size=(img.shape[1], img.shape[0]),
                    score_threshold=0.9, 
                    nms_threshold=0.3,
                    top_k=5000
                )
                faces = face_detector.detect(img)
                # print(faces)
                # print(f"proccessing folder{current_directory[-1]}\tImage{path}")
                if faces[1] is not None:
                    for face in faces[1]:
                        x, y, w, h = face[0:4].astype(int)
                        # print(f"height: {h}\nwidth: {w}")
                        # grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        normalized_image = cv2.normalize(
                            img, 
                            None, 
                            alpha=0, 
                            beta=255, 
                            norm_type=cv2.NORM_MINMAX
                        )
                        cv2.rectangle(normalized_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        if h>30 and w>30 and path!="1.jpg":

                            roi_color = normalized_image[y:y + h, x:x + w] 
                            try:
                                resized_image=resize_image(roi_color, (150, 150))
                                # print(f"Hi there. I detected Folder{current_directory[-1]},\t{path}")
                                cv2.imwrite(f"C:\\Users\\Sepehr\\Desktop\\project1\\detected faces\\{current_directory[-1]}\\{path}", resized_image)
                            except Exception as e:
                                print(e)
                        if path == "1.jpg":
                            roi_color = img[y:y + h, x:x + w]
                            try:
                                resized_image=resize_image(roi_color, (150, 150))
                                # print(f"Hi there. I detected Folder{current_directory[-1]},\t{path}")
                                cv2.imwrite(f"C:\\Users\\Sepehr\\Desktop\\project1\\detected faces\\{current_directory[-1]}\\{path}", resized_image)
                            except Exception as e:
                                print(f"Folder {current_directory}\tImage{path} NO FACE detected")
                                print(e)
                else:
                    print(f"Folder {current_directory}\tImage{path} NO FACE detected")


'''
Verify
Pictures:
'''
# face detection model for my facenet
# mtcnn = MTCNN()

#face recog model
resnet = InceptionResnetV1(pretrained="vggface2").eval()


detected_images_root = r"C:\Users\Sepehr\Desktop\project1\detected faces"
cnt = 0
for root, dirs, files in os.walk(detected_images_root, topdown=False):
     for name in dirs:
        current_directory = os.path.join(root, name)
        # the parameter "flag" is for setting a manual anchor (True -> manual)
        flag = False
        # print(current_directory)
        if not flag:
            image_anchor = Image.open(f"{current_directory}\\1.jpg")
        else:
            image_anchor = Image.open("C:\\Users\\Sepehr\\Desktop\\project1\\detected faces\\1\\1.jpg")
            #setting up some changes on raw files. (Denoise, Grayscale, Contrast, and Normalize illumination.)
            normalized_imageAnch = normalize_illumination(image_anchor)
            image_anchor_gry = ImageOps.grayscale(image_anchor)
            denoised_image_anch = image_anchor_gry.filter(ImageFilter.MedianFilter(size=3))
            enhancer_anch = ImageEnhance.Contrast(denoised_image_anch)
            image_anchor = grayscale_to_3channels(enhancer_anch.enhance(2))


        transform = transforms.ToTensor()
        image_anchor = ImageEnhance.Contrast(image_anchor)
        image_anchor = image_anchor.enhance(1)

        # tensor_imgAnchor = transform(grayscale_to_3channels(image_anchor))
        tensor_imgAnchor = transform(image_anchor)
        # faces1, _ = mtcnn.detect(image_anchor)
        for path in os.listdir(current_directory):
            if path != "1.jpg":
                image_data = Image.open(os.path.join(current_directory, path))
                #setting up some changes on raw files. (Denoise, Grayscale, Contrast, and Normalize illumination.)
                image_data_gryscale = ImageOps.grayscale(image_data)
                normalized_image = normalize_illumination(image_data_gryscale)
                denoised_image_data = normalized_image.filter(ImageFilter.MedianFilter(size=3))
                # upsampledImg2 = sr.upsample(np.array(image_data))
                # im2 = ImageOps.grayscale(Image.fromarray(upsampledImg2)) 
                # norm2 = ImageOps.equalize(im2)
                # rgb_image_data = grayscale_to_3channels(im2)
                enhancer_data = ImageEnhance.Contrast(denoised_image_data)
                enhanced_imageData = enhancer_data.enhance(2)
                
                # denoised_image_data = image_data_gryscale.filter(ImageFilter.MedianFilter(size=3))
                # enhanced_imageData.save(f"C:\Users\Sepehr\Desktop\project1\out\{path}.jpg")
                # faces2, _ = mtcnn.detect(image_data)
                # aligned_anchor = mtcnn(image_anchor)
                # aligned_data = mtcnn(image_data)
                tensor_imgData = transform(grayscale_to_3channels(enhanced_imageData))
                stupidImage = grayscale_to_3channels(image_data_gryscale)
                stupidImage.save(f"C:\\Users\\Sepehr\\Desktop\\project1\\out\\{cnt+1}.jpg")
                embeddings_anchor = resnet(tensor_imgAnchor.unsqueeze(0)).detach()
                embeddings_data = resnet(tensor_imgData.unsqueeze(0)).detach()
                distance = (embeddings_anchor - embeddings_data).norm().item()
                print(f"embeding's distance {current_directory[-1]}\{path} from  anchor: {distance}, verif={distance<0.91}")
                cnt+=1

                # image_data = open(os.path.join(current_directory, path), "rb").read()
                # response = requests.post("http://localhost:80/v1/vision/face/match",files={"image1":image_anchor,"image2":image_data}, data={"min_confidence":0.40}).json()

                # if response['success'] and not flag:
                #     # image_anchor = image_data
                #     flag = True
                #     anchors.append(f"{current_directory[-1]}\{path}")
                # print(f"Verified:{response}, for pic:{current_directory[-1]}\{path}, anch:{current_directory[-1]}\\1jpg")

         

# pic4root = r"C:\Users\Sepehr\Desktop\project1\Images\5"
# image_data1 = open("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\5\\2.jpg","rb").read()
# print(type(image_data1))

# for root, dirs, files in os.walk(pic4root, topdown=False):
#     for file in files:
#         direct = os.path.join(pic4root, file)
#         image_data2 = open(direct,"rb").read()
#         response = requests.post("http://localhost:80/v1/vision/face/match",files={"image1":image_data1,"image2":image_data2}, data={"min_confidence":0.10}).json()
#         print(f"Verified:{response}, for pic:{file}")

# # print(f"Verified:{response['success']}")
# image_data1 = open("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\5.jpg","rb").read()
# image_data2 = open("C:\\Users\\Sepehr\\Desktop\\project1\\Images\\1\\1.jpg","rb").read()
# response = requests.post("http://localhost:80/v1/vision/face/match",files={"image1":image_data1,"image2":image_data2}).json()
# print(response)


