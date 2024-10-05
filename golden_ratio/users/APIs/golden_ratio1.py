import os
import json
import mediapipe as mp
import numpy as np
from landmarks import l1
from refer import l2,rft_arr,rft1_arr
from PIL import Image
import cv2


folder_path = 'C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject'

r1=rft_arr
r2=rft1_arr

# Define landmark filter with two pairs
filter_landmark1 = l1
filter_landmark2=l2
# Define paths and filenames


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
ft_arr=[]
ft1_arr=[]
Name=['Distance between the eyes',
'Width of right eye ',
'Width of left eye ',
'End of the arc to length of the right eyebrow',
'End of the arc to length of the left eyebrow',
'Right mouth edge to side of the face ',
'Left mouth edge to side of the face',
'Center of mouth to chin',
'Width of upper lip',
'Width of the nose',
'Width of forehead',
'Width of the chin']
def idraw_lines_with_text(image, landmarks, landmark_pairs):
    sum1 = 0
    red = []
    reference_real_world_size = 35.0
    for pair, n ,name in zip(landmark_pairs,r1,Name):
        start_idx = pair['start']
        end_idx = pair['end']
        reference=n
        #label = pair['label']
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        #print(distance)
        dis=str(distance)
        ft=float(dis[0:3])
        ft_arr.append(ft)
        v =  ft/reference
        if v*100 >100:
            v = reference/ft
            #print(f"Distance from {start_idx} to {end_idx}: {ft}mm,  values:\033[31m{v*100}\033[0m")
            print(f"{name}\npatient:{ft}mm\treference:{n}\nGR_percentage:\033[31m{v* 100}\033[0m")
        else:
            #print(f"Distance from {start_idx} to {end_idx}: {ft}mm, values:\033[31m{v * 100}\033[0m")
            print(f"{name}\npatient:{ft}mm\treference:{n}\nGR_percentage:\033[31m{v*100}\033[0m")
        sum1 = sum1 + (v*100)
        red.append(v*100)
       # print(v * 100)

        #Draw the line
        cv2.line(image, start_pt, end_pt, (0, 0, 255, 0), 2)

        # Calculate the midpoint
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

        # Adjust the position of the text to be above the line
        text_position = (midpoint[0], midpoint[1] - 10)
        #Annotate with the label
        cv2.putText(image, str(v*100), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return image ,sum1, red

# value 1.618

Name2=['Distance from right eye inner edge to side of face',
'Distance from left eye inner edge to side of face',
'Width of the right eyebrow',
'Width of the left eyebrow',
'Length of the nose',
'Width of the mouth',
'Starting of the nose to center of the mouth',
'Width of lower lip',
'Right eye inner edge to cheekbone',
'Left eye inner edge to cheekbone',
'Centre of forehead to right side of face',
'Centre of forehead to left side of face' ]
def idraw_lines_with_text1(image, landmarks, landmark_pairs):
    sum2 = 0
    green = []
    reference_real_world_size = 3.5
    for pair ,r  ,name2 in zip(landmark_pairs,r2,Name2):

        start_idx = pair['start']
        end_idx = pair['end']
        #reference = pair['refval']
        reference=r
        #print(reference)
        #label = pair['label']
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        #print(distance)
        dis=str(distance)
        ft1 = float(dis)
        ft1_arr.append(ft1)


        c = reference/ 1.618
        s = ft1/c


        s1=s/1.618

        if s1*100>100:
            s1=1.618/s
            #print(s1)
            print(f"{name2}\npatient:{ft1}mm\treference:{r}\nGR_percentage:\033[32m{s1*100}\033[0m")
        else:

            print(f"{name2}\npatient:{ft1}mm\treference:{r}\nGR_percentage:\033[32m{s1 * 100}\033[0m")
            # print(s1)

        #(s * 100)
        sum2 = sum2 +(s1*100)
        green.append(s1*100)
        #Draw the line
        cv2.line(image, tuple(start_pt),tuple(end_pt), (0, 255, 0), 2)

        # Calculate the midpoint
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

        # Adjust the position of the text to be above the line
        text_position = (midpoint[0], midpoint[1] - 10)

        #Annotate with the label
        cv2.putText(image, str(s1*100), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, ( 0, 255,0,0), 1)




    return image,sum2,green

def apply_filter(image, landmarks, landmark_indices):
    # Draw circles around the specified landmark indices
    for idx in landmark_indices:
        landmark_pt = tuple(map(int,landmarks[idx]))
        cv2.circle(image, landmark_pt, 3, (0, 255, 0), -1)
    # for idx in landmark_indices1:
    #     landmark_pt = landmarks[idx]
    #     cv2.circle(image, landmark_pt, 3, (0,0, 255, 0), -1)
    return image

#Load the input image
image_path1 = "C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/app_folder/input_image/im37.jpg"
#image_path1='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/golden ratio/output.jpg'
image_path2 = os.path.join(folder_path, 'modified_image_shrink.jpg')

image = cv2.imread(image_path1)
if image is None:
    raise FileNotFoundError(f"Failed to load image from: {image_path1}")

results = face_mesh.process(image)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
         # Convert normalized landmarks to pixel coordinates
         height, width, _ = image.shape
         l= []
         landmarks = [(int(landmark.x * width),int(landmark.y * height)) for landmark in face_landmarks.landmark]

         #Apply filter to the detected landmarks using filter_landmark indices
         landmark_indices= [lm['start'] for lm in filter_landmark1] + [lm['end'] for lm in filter_landmark1]
         image = apply_filter(image, landmarks, landmark_indices)
         landmark_indices1 = [lm['start'] for lm in filter_landmark2] + [lm['end'] for lm in filter_landmark2]
         image1 = apply_filter(image, landmarks, landmark_indices1)
         image,rsum,redlist = idraw_lines_with_text(image, landmarks, filter_landmark1)
         image1, gsum,greenlist = idraw_lines_with_text1(image1, landmarks, filter_landmark2)
         #print(ft_arr,ft1_arr)
         asum = rsum + gsum
         avg = asum/(len(l1) + len(l2))
     
         for landmark_pt in landmarks:
             cv2.circle(image, landmark_pt, 1, (255, 0, 0), -1)  # Draw a small dot for each landmark

print(len(rft_arr),len(rft1_arr))
#print(ft_arr,ft1_arr)
output_path = os.path.join(folder_path, 'input_marked_image.jpg')
cv2.imwrite(output_path, image)
# Display the modified image
cv2.imshow("Filtered_Image1", image)
cv2.waitKey(0)
cv2.destroyAllWindows()