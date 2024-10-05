import os
import json
import mediapipe as mp
import numpy as np
from m2landmark import l1
from m2rlandmarks import l2
from PIL import Image
import cv2
import pandas as pd

folder_path = 'C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject'
folder_path1='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/golden ratio'

# Define landmark filter with two pairs
filter_landmark3 =l1
filter_landmark4=l2


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


# Function to apply a filter to the detected landmarks

rft_arr = []
rft1_arr = []


#value1
def rdraw_lines_with_text(image, landmarks, landmark_pairs):
    sum=0


    reference_real_world_size = 3.5
    for pair in landmark_pairs:
        start_idx = pair['start']
        end_idx = pair['end']
        #label = pair['label']

        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        dis=str(distance)
        rft=float(dis[0:3])
        rft_arr.append(rft)



        #rdis.append(rft)
        #print(f"Distance from {start_idx} to {end_idx}: {rft}mm")
       # print(v * 100)

        #Draw the line
        cv2.line(image, start_pt, end_pt, (0, 0, 255, 0), 2)

        # Calculate the midpoint
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

        # Adjust the position of the text to be above the line
        text_position = (midpoint[0], midpoint[1] - 10)

        #Annotate with the label
        #cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

# value 1.618

def rdraw_lines_with_text1(image, landmarks, landmark_pairs):
    reference_real_world_size = 3.5

    for pair in landmark_pairs:
        start_idx = pair['start']
        end_idx = pair['end']
        #label = pair['label']
        start_pt =landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        dis=str(distance)
        rft1 = float(dis[0:3])
        rft1_arr.append(rft1)
        #rdis1.append(rft1)
        #print(f"Distance from {start_idx} to {end_idx}: {rft1}mm")
        #(s * 100)


        #Draw the line
        cv2.line(image, (start_pt), (end_pt), (0, 255, 0), 2)

        # Calculate the midpoint
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

        # Adjust the position of the text to be above the line
        text_position = (midpoint[0], midpoint[1] - 10)

        #Annotate with the label
       # cv2.putText(image, str(w), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image


def apply_filter(image, landmarks, landmark_indices):
    # Draw circles around the specified landmark indices
    for idx in landmark_indices:
        landmark_pt = landmarks[idx]
        cv2.circle(image, landmark_pt, 3, (0, 255, 0), -1)
    # for idx in landmark_indices1:
    #     landmark_pt = landmarks[idx]
    #     cv2.circle(image, landmark_pt, 3, (0,0, 255, 0), -1)
    return image

def image_per(image_path_var):
    image = cv2.imread(image_path_var)
    # s_image = cv2.imread(s_image_path)
    # b_image = cv2.imread(b_image_path)

    if image is None:
       raise FileNotFoundError(f"Failed to load image from: {image_path_var}")


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with MediaPipe face mesh
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          height, width, _ = image.shape
          l= []
          landmarks = [(int(landmark.x * width),int(landmark.y * height)) for landmark in face_landmarks.landmark]


          landmark_indices = [lm['start'] for lm in filter_landmark3] + [lm['end'] for lm in filter_landmark3]
          image = apply_filter(image, landmarks, landmark_indices)
          landmark_indices1 = [lm['start'] for lm in filter_landmark4] + [lm['end'] for lm in filter_landmark4]
          image = apply_filter(image, landmarks, landmark_indices1)


          image = rdraw_lines_with_text(image, landmarks, filter_landmark3)
          image= rdraw_lines_with_text1(image, landmarks, filter_landmark4)

          #for landmark_pt in landmarks:
              #cv2.circle(image,landmark_pt, 1, (255, 0, 0), -1)  # Draw a small dot for each landmark
    return image,rft_arr,rft1_arr


image_path1 = os.path.join(folder_path1,'Frame Resized.jpg')
# s_image_path = os.path.join(folder_path1, 'modified_image_bulge.jpg')
# b_image_path = os.path.join(folder_path1, 'modified_image_shrink.jpg')
image=cv2.imread(image_path1)
# input_im=image_per(image_path1)
# print(input_im)
input_image, rft_arr, rft1_arr = image_per(image_path1)
print(rft_arr)
print(rft1_arr)
cv2.imshow("Filtered_Image", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


