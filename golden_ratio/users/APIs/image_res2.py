import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import matplotlib.pyplot as plt
import os
from image import inpt_image, ref_shrink_image, ref_bulge_image
import math
import numpy as np
import cv2 as cv
import mediapipe as mp
from PIL import Image


folder_path='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/app_folder'
#folder_path_inpt='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/golden ratio'


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
  folder_path_task = 'C:/Users/user/PycharmProjects/goldebn ratio/face_landmarker.task'
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




#initialize input image, reference image for shrink and reference image for bulge and store their coordinates(x, y, z) as json file

im= inpt_image
#img = cv2.imread(im)
input_image=get_image_coordinates(im)

im1=ref_shrink_image
#im1= os.path.join(folder_path_ref,'_com4.jpg')
#img = cv2.imread(im)
ref_image_shrink=get_image_coordinates(im1)

im2=ref_bulge_image
#im2= os.path.join(folder_path_inpt,'im79.jpg')
#img1 = cv2.imread(im1)
ref_image_bulge=get_image_coordinates(im2)



#json data for reference image for shrink
json_formatted_str = json.dumps(ref_image_shrink, indent=2)
with open(os.path.join(folder_path,'ref_coordinates_shrink.json'), 'w') as json_file1:
   json_file1.write(json_formatted_str)

#json data for reference image for bulge
json_formatted_str1 = json.dumps(ref_image_bulge, indent=2)
with open(os.path.join(folder_path,'ref_coordinates_bulge.json'), 'w') as json_file2:
    json_file2.write(json_formatted_str1)

#json data for reference image for input
json_formatted_str2 = json.dumps(input_image, indent=2)
with open(os.path.join(folder_path,'input_coordinates.json'), 'w') as json_file:
  json_file.write(json_formatted_str2)















