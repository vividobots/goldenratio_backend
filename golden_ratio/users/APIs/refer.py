import os
import json
import mediapipe as mp
import numpy as np

from .landmarks import l1
from .landmarks2 import l2
from PIL import Image
import cv2
import pandas as pd


# Define paths and filenames
folder_path = '/home/ubuntu/goldenratio_backend/Simage'
# folder_path1='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/golden ratio'


# Define landmark filter with two pairs
filter_landmark3 =l1


filter_landmark4= l2
mp_face_mesh = mp.solutions.face_mesh
#face_mesh = mp_face_mesh.FaceMesh()
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Function to apply a filter to the detected landmarks
rdis=[]

rdis1=[]

rft_arr = []
rft1_arr=[]
#print(l1[0])
#value1
def rdraw_lines_with_text(image, landmarks, landmark_pairs):
    reference_real_world_size = 3.5
    for pair in landmark_pairs:
        start_idx = pair['start']
        end_idx = pair['end']
        #label = pair['label']

        start_pt = tuple(map(int,landmarks[start_idx]))
        end_pt = tuple(map(int,landmarks[end_idx]))
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        dis=str(distance)
        rft=float(dis[0:3])
        rft_arr.append(rft)
        rdis.append(rft)
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

    return rdis
# value 1.618
def rdraw_lines_with_text1(image, landmarks, landmark_pairs):
    reference_real_world_size = 3.5
    for pair in landmark_pairs:
        start_idx = pair['start']
        end_idx = pair['end']
        #label = pair['label']
        start_pt =tuple(map(int, landmarks[start_idx]))
        end_pt = tuple(map(int,landmarks[end_idx]))
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        dis=str(distance)
        rft1 = float(dis[0:3])
        rft1_arr.append(rft1)
        rdis1.append(rft1)
        #print(f"Distance from {start_idx} to {end_idx}: {rft1}mm")
        #(s * 100)


        #Draw the line
        cv2.line(image, tuple(start_pt), tuple(end_pt), (0, 255, 0), 2)

        # Calculate the midpoint
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

        # Adjust the position of the text to be above the line
        text_position = (midpoint[0], midpoint[1] - 10)

        #Annotate with the label
        #cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return rdis1

def apply_filter(image, landmarks, landmark_indices):
    # Draw circles around the specified landmark indices
    for idx in landmark_indices:
        landmark_pt = landmarks[idx]
        cv2.circle(image, landmark_pt, 3, (0, 255, 0), -1)
    # for idx in landmark_indices1:
    #     landmark_pt = landmarks[idx]
    #     cv2.circle(image, landmark_pt, 3, (0,0, 255, 0), -1)
    return image

# Load the input image
image_path = os.path.join(folder_path, 'aj2.jpg')
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Failed to load image from: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with MediaPipe face mesh
results = face_mesh.process(image_rgb)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Convert normalized landmarks to pixel coordinates
        height, width, _ = image.shape
        l= []
        landmarks = [(int(landmark.x * width),int(landmark.y * height)) for landmark in face_landmarks.landmark]

         # Apply filter to the detected landmarks using filter_landmark indices
        landmark_indices = [lm['start'] for lm in filter_landmark3] + [lm['end'] for lm in filter_landmark3]
        image = apply_filter(image, landmarks, landmark_indices)
        landmark_indices1 = [lm['start'] for lm in filter_landmark4] + [lm['end'] for lm in filter_landmark4]
        image1 = apply_filter(image, landmarks, landmark_indices1)


        rdis = rdraw_lines_with_text(image, landmarks, filter_landmark3)
        rdis1 = rdraw_lines_with_text1(image1, landmarks, filter_landmark4)
        # Draw circles around all landmarks (for debugging purposes)
        # for landmark_pt in landmarks:
        #     cv2.circle(image,tuple(map(int, landmark_pt)), 1, (255, 0, 0), -1)  # Draw a small dot for each landmark

print(rft_arr)
print(rft1_arr)
    # print("________________________________________________________________________________________________")
# print(rft1_arr)
dfrft=pd.DataFrame(rft_arr)
dfrft1=pd.DataFrame(rft1_arr)
dfrft.to_csv("l1.csv")
dfrft1.to_csv("l2.csv")


#from landmarks2 import l2 as nl2
# for i in range(0,len(filter_landmark3)):

#     tempdict = face_landmarks[i]
#     tempdict["refval"] = rft_arr[i]
#     face_landmarks[i] = tempdict


# #from landmarks2 import l2 as nl2
# for j in range(0,len(filter_landmark4)):
#     tempdist1 = filter_landmark4[j]
#     tempdist1["refval"] = rft1_arr[j]
#     filter_landmark4[j] = tempdist1


#print(l1)
output_path = os.path.join(folder_path, 'input_marked_image.jpg')
cv2.imwrite(output_path, image)
# Display the modified image
#cv2.imshow("Filtered_Image", image)
#cv2.imshow("Filtered_Image", image1)

cv2.waitKey(0)
cv2.destroyAllWindows()

