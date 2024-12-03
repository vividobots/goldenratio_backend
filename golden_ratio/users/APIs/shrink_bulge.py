import shape
from flask import Flask
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import matplotlib.pyplot as plt
import os
# from image import inpt_image, ref_shrink_image, ref_bulge_image
import math
import numpy as np
import cv2 as cv
import mediapipe as mp
from PIL import Image
from  demo2 import filter_landmark_shrink as filter_landmark_s
from  demo2 import filter_landmark_bulge as filter_landmark_b
# from shape import predict_shape_face , classes

import sys
# from models import UploadedImage
import numpy as np
import matplotlib.pyplot as plt


# import torch
# import torchvision
# import torch.nn as nn
# from torchvision import transforms as T

# from PIL import ImageFile
# from keras.preprocessing.image import load_img,img_to_array
# ImageFile.LOAD_TRUNCATED_IMAGES=True



sb_ratios=[[],[]]
# app = Flask(__name__)
input_image = sys.argv[1]
Id=sys.argv[2]



print("123---->",input_image)
# inpt_image='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/app_folder/input_image/im78.jpg'
ref_shrink_image="D:/goldenratio_backend/reference_images/_com4.jpg"
ref_bulge_image="D:/goldenratio_backend/origional_images/im77.jpg"
folder='shrink_bulge_output/'

input_path = r'/input_image/'
f_img = input_image
# f_img = os.path.join(input_path,input_image)
print("f_img----------------->",f_img)


def get_image_coordinates(im):
  def plot_face_blendshapes_bar_graph(face_blendshapes):

    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores,
                  label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
      plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")
    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()

  #plt.show()
  folder_path_task = 'D:/goldenratio_backend/face_landmarker.task'
  base_options = python.BaseOptions(model_asset_path=folder_path_task)
  options = vision.FaceLandmarkerOptions(base_options=base_options,
  output_face_blendshapes=True,
  output_facial_transformation_matrixes=True,
  num_faces=1)
  detector = vision.FaceLandmarker.create_from_options(options)
  # STEP 3: Load the input image.


  image = mp.Image.create_from_file(im)

  detection_result = detector.detect(image)


  plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

  val = detection_result.face_landmarks
  no =0
  print(len(val))
  output_data = [[]]

  output_data[0] = [
  [
  {
  "x": float(item.x),
  "y": float(item.y),
  "z": item.z,
  }
  for index, item in enumerate(category_list)
  ]
  for no, category_list in enumerate(detection_result.face_landmarks)
  ]
  return output_data



im= input_image
input_image=get_image_coordinates(im)

im1=ref_shrink_image
ref_image_shrink=get_image_coordinates(im1)

im2=ref_bulge_image
ref_image_bulge=get_image_coordinates(im2)
print("I",input_image)



input_shrink_values = []

def input_distance_coordinates_shrink():
    for item in input_image[0]:
        for landmark in filter_landmark_s:
            id_value = item[landmark['start']]
            id_value1 = item[landmark['end']]
            landmark1 = (id_value['x'], id_value['y'], id_value['z'])
            landmark2 = (id_value1['x'], id_value1['y'], id_value1['z'])
            distance1 = np.linalg.norm(np.array(landmark2) - np.array(landmark1))
            values = distance1

            input_shrink_values.append(values)
    return input_shrink_values

s = input_distance_coordinates_shrink()
print(len(s))
print(s)

input_bulge_values = []
def input_distance_coordinates_bulge():
    for item in input_image[0]:
        for landmark in filter_landmark_b:
            id_value = item[landmark['start']]
            id_value1 = item[landmark['end']]
            landmark1 = (id_value['x'], id_value['y'], id_value['z'])
            landmark2 = (id_value1['x'], id_value1['y'], id_value1['z'])
            distance1 = np.linalg.norm(np.array(landmark2) - np.array(landmark1))
            values = distance1
            input_bulge_values.append(values)

    return input_bulge_values

b = input_distance_coordinates_bulge()
print(len(b))
print(b)

ref_shrink_values = []
ref_bulge_values = []

def reference_distance_coordinates_shrink():
  for item in ref_image_shrink[0]:
    for landmark in filter_landmark_s:
      id_value = item[landmark['start']]
      id_value1 = item[landmark['end']]
      landmark1 = (id_value['x'], id_value['y'], id_value['z'])
      landmark2 = (id_value1['x'], id_value1['y'], id_value1['z'])
      distance = np.linalg.norm(np.array(landmark2) - np.array(landmark1))
      values = distance
      ref_shrink_values.append(values)
  return ref_shrink_values


rs = reference_distance_coordinates_shrink()
print(len(rs))
print("REFSHRK:",rs)


def reference_distance_coordinates_bulge():
  for item in ref_image_bulge[0]:
    for landmark in filter_landmark_b:
      id_value = item[landmark['start']]
      id_value1 = item[landmark['end']]
      landmark1 = (id_value['x'], id_value['y'], id_value['z'])
      landmark2 = (id_value1['x'], id_value1['y'], id_value1['z'])
      distance = np.linalg.norm(np.array(landmark2) - np.array(landmark1))
      values = distance
      ref_bulge_values.append(values)
  return ref_bulge_values


rb = reference_distance_coordinates_bulge()
print(len(rb))
print("REFBLG:",rb)

input_ref_diffrence_shrink = []
input_ref_diffrence_bulge = []

shrink_amount = []
bulge_amount = []


def input_reference_difference_shrink():
  print(ref_shrink_values)
  print(input_shrink_values)

  for landmark, i, j in zip(filter_landmark_s, input_shrink_values, ref_shrink_values):
    s = j - i
    d = i + s
    if not str(d).startswith('0.0'):
      u = (d) * 10 ** -1
      values = {'l': landmark['end'], 'amount': u}
      input_ref_diffrence_shrink.append(u)
      shrink_amount.append(values)
    else:
      u = (d) * 10 ** -1
      values = {'l': landmark['end'], 'amount': u}
      input_ref_diffrence_shrink.append(s)
      shrink_amount.append(values)


  return shrink_amount
  # return input_ref_diffrence_shrink


irs = input_reference_difference_shrink()
print("R",irs)
#
#
def input_reference_difference_bulge():
  print(ref_bulge_values)
  print(input_bulge_values)
  for landmark, i, j in zip(filter_landmark_b, input_bulge_values, ref_bulge_values):
    s = j - i
    d = i - s
    print(d)
    if not str(d).startswith('0.0'):
      d = (d) * 10 ** -1

      values = {'l': landmark['end'], 'amount': -d}
      input_ref_diffrence_bulge.append(d)
      bulge_amount.append(values)
    else:
      d = (d) * 10 ** -1
      values = {'l': landmark['end'], 'amount': -d}
      input_ref_diffrence_bulge.append(s)
      bulge_amount.append(values)

  return bulge_amount
  # return input_ref_diffrence_bulge


irb = input_reference_difference_bulge()
print(irb)



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
# f_img = input_image

print("1233445----->",f_img)
im_cv = cv.imread(f_img)

(h, w, _) = im_cv.shape
scale_x = 1.0
scale_y = 1.0
radius = min(w, h) / 2

flex_x = np.zeros((h, w), np.float32)
flex_y = np.zeros((h, w), np.float32)

results = face_mesh.process(im_cv)
landmarks_data = {}


def shrink(im_cv):
  if results.multi_face_landmarks:
    my_list = shrink_amount

  for element in my_list:
    landmark_index = element['l']

    amount = element['amount']
    landmark_point = results.multi_face_landmarks[0].landmark[landmark_index]
    # print(amount)
    center_x = int(landmark_point.x * im_cv.shape[1])
    center_y = int(landmark_point.y * im_cv.shape[0])

    for y in range(h):
      delta_y = scale_y * (y - center_y)
      for x in range(w):
        delta_x = scale_x * (x - center_x)
        distance = delta_x * delta_x + delta_y * delta_y
        if distance >= (radius * radius):
          flex_x[y, x] = x
          flex_y[y, x] = y
        else:
          factor = 1.0
          if distance > 0.0:
            factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
          flex_x[y, x] = factor * delta_x / scale_x + center_x
          flex_y[y, x] = factor * delta_y / scale_y + center_y

    dst = cv.remap(im_cv, flex_x, flex_y, cv.INTER_LINEAR)
    im_cv = dst
  return im_cv

def bulge(im_cv):
  if results.multi_face_landmarks:
    my_list = bulge_amount

  for element in my_list:
    landmark_index = element['l']

    amount = element['amount']
    landmark_point = results.multi_face_landmarks[0].landmark[landmark_index]
    # print(amount)
    center_x = int(landmark_point.x * im_cv.shape[1])
    center_y = int(landmark_point.y * im_cv.shape[0])

    for y in range(h):
      delta_y = scale_y * (y - center_y)
      for x in range(w):
        delta_x = scale_x * (x - center_x)
        distance = delta_x * delta_x + delta_y * delta_y
        if distance >= (radius * radius):
          flex_x[y, x] = x
          flex_y[y, x] = y
        else:
          factor = 1.0
          if distance > 0.0:
            factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
          flex_x[y, x] = factor * delta_x / scale_x + center_x
          flex_y[y, x] = factor * delta_y / scale_y + center_y

    dst = cv.remap(im_cv, flex_x, flex_y, cv.INTER_LINEAR)
    im_cv = dst
  return im_cv

# print("shrink/ bulge ")
output_file_shrink = os.path.join(folder,'shrink'+Id+'.jpg')
output_file_bulge = os.path.join(folder, 'bulge'+Id+'.jpg')
sb_json='SHRK_BLG_json/'
json_path = os.path.join(sb_json,f'sb_{Id}.json')
print("JSON->",json_path)

# filepath = UploadedImage.objects.get(id=Id)
# file_path.shrink_bulge_json=json_path
# sb_ratios[1].append({
#   'shrink':output_file_shrink
# })
# sb_ratios[2].append({
#   'bulge':output_file_bulge
# })

# file_path.save()

# dst_s = shrink(im_cv)
# dst_b = bulge(im_cv)

# def out(dst):
#   cv.imwrite(output_file_shrink, dst_s)
#   ks = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#   shrink_shape = cv.filter2D(src=dst_s, ddepth=-1, kernel=ks)
#   return 


dst_s = shrink(im_cv)
cv.imwrite(output_file_shrink, dst_s)
ks = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
shrink_shape = cv.filter2D(src=dst_s, ddepth=-1, kernel=ks)

dst_b = bulge(im_cv)
cv.imwrite(output_file_bulge, dst_b)
ks = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
bulge_shape = cv.filter2D(src=dst_b, ddepth=-1, kernel=ks)

# cv.imshow('image_original', im_cv)
# cv.imshow('image_shrink', shrink_shape)
# cv.imshow('image_bulge', bulge_shape)
# cv.waitKey(0)
# cv.destroyAllWindows()


#SHAPE DETECTION FUNCTION--------------------------------------------------------------------------------------------
# shape_mapping = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

# def predict_face_shape(image_path):
  
#     num_classes=5
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     transform = T.Compose([
#         T.Resize((224, 224)),  # Resize image to match the input size of the model
#         T.ToTensor(),  # Convert to Tensor
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like in training
#     ])
    
#     model = torchvision.models.efficientnet_b4(pretrained=True)
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.3, inplace=True),
#         nn.Linear(model.classifier[1].in_features, num_classes)  # Change the output layer to match your classes
#     )
#     model_path = 'best_model.pth'
#     model.load_state_dict(torch.load(model_path,map_location=device))
#     model.eval()



#     image = Image.open(image_path).convert('RGB')  # Open and convert image to RGB
#     image = transform(image).unsqueeze(0)  # Apply transformation and add batch dimension

#     # Check if GPU is available and move the model and data to GPU if possible
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     image = image.to(device)

#     # Make prediction
#     with torch.no_grad():  # No need to track gradients for inference
#         output = model(image)  # Get model output
#         _, predicted_class = torch.max(output, 1)  # Get the predicted class index

#     # Map the predicted class index to the shape label
#     predicted_label = shape_mapping[predicted_class.item()]
    
#     return predicted_label

# predicted_shape = predict_face_shape(input_image)


# if predicted_shape==shape_mapping[0] or predicted_shape==shape_mapping[1] or predicted_shape==shape_mapping[4]:
#     print("0987665")
    
#     dst_s = shrink(im_cv)
#     cv.imwrite(output_file_shrink, dst_s)
#     ks = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#     shrink_shape = cv.filter2D(src=dst_s, ddepth=-1, kernel=ks)
#     cv.imshow('image_original', im_cv)
#     cv.imshow('image_shrink', shrink_shape)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# elif predicted_shape==shape_mapping[2] or predicted_shape==shape_mapping[3]:
#     print("1111111111")
#     dst_b = bulge(im_cv)
#     cv.imwrite(output_file_bulge, dst_b)
#     ks = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#     bulge_shape = cv.filter2D(src=dst_b, ddepth=-1, kernel=ks)
#     cv.imshow('image_original', im_cv)
#     cv.imshow('image_bulge', bulge_shape)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#---------------------------------------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     app.run(debug=True,port=8001)









