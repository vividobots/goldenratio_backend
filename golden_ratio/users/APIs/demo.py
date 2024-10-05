# import os

# from models import User
# import cv2
# import numpy as np
# import mediapipe as mp
# from io import BytesIO

# from landmarks import l1
# from m2r import filter_landmark3 as filter_landmark3,filter_landmark4 as filter_landmark4
# from refer import l2, rft_arr, rft1_arr
# from phiweb import phi_matrix_method
# from sym_asym import final_res


# Name = ['Eye Spacing', 'Right eye width', 'Left eye width',
#         'start of brow to arc', 'Left start of brow to arc',
#         'Right oral corner to face side', 'Left oral corner to face side',
#         'Oral center to chin', 'Upper lip width', 'Nose width', 'Forehead width',
#         'ChRightin width']

# Name2 = ['Right eye corner to face edge', 'Left eye corner to face edge',
#          'Right eyebrow width', 'Left eyebrow width', 'Nose length',
#          'Oral width', 'Nose tip to oral center', 'Lower lip width',
#          'Right eye corner to cheekbone', 'Left eye corner to cheekbone',
#          'Middle forehead to right face edge', 'Middle forehead to left face edge']



# def idraw_lines_with_text(image, landmarks, landmark_pairs):
#     sum1 = 0
#     red = []
#     reference_real_world_size = 3.5
#     distances = []  # List to store distances (ft)
#     for pair, n, name in zip(landmark_pairs, rft_arr, Name):
#         start_idx = pair['start']
#         end_idx = pair['end']
#         reference = pair['refval'] #n is aso used for reference both pair['refval'] and n for same list  rft_arr
#         start_pt = landmarks[start_idx]
#         end_pt = landmarks[end_idx]
#         distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt)) / reference_real_world_size
#         ft = float(distance)  # Store the distance as float
#         distances.append(ft)  # Append the distance to the list
#         v = ft / reference
#         s1=v/1.618
#         if v * 100 > 100:
#             s1=1.618/v
#             v = reference / ft
#             print(f"{name}\npatient:{ft:.2f}\treference:{n}\nGR_percentage:{s1 * 100:.2f}")
#         else:
#             print(f"{name}\npatient:{ft:.2f}\treference:{n}\nGR_percentage:{s1 * 100:.2f}")
#         sum1 += v * 100
#         red.append(v * 100)
#         cv2.line(image, start_pt, end_pt, (0, 0, 255, 0), 2)
#         midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
#         text_position = (midpoint[0], midpoint[1] - 10)
#         cv2.putText(image, f"{v * 100}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
#     return image, sum1, red, distances  # Return the list of distances


# def idraw_lines_with_text1(image, landmarks, landmark_pairs):
#     sum2 = 0
#     green = []
#     reference_real_world_size = 3.5
#     distances = []  # List to store distances (ft1)
#     for pair, r, name2 in zip(landmark_pairs, rft1_arr, Name2):
#         start_idx = pair['start']
#         end_idx = pair['end']
#         reference = pair['refval'] #r is also used for 'reference' both pair['refval'] and n comes from same list  rft1_arr
#         start_pt = landmarks[start_idx]
#         end_pt = landmarks[end_idx]
#         distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt)) / reference_real_world_size
#         ft1 = float(distance)  # Store the distance as float
#         distances.append(ft1)  # Append the distance to the list
#         c = reference / 1.618
#         s = ft1 / c
#         s1 = s / 1.618
#         if s1 * 100 > 100:
#             s1 = 1.618 / s
#             print(f"{name2}\npatient:{ft1:.2f}\treference:{r}\nGR_percentage:{s1 * 100:.2f}")
#         else:
#             print(f"{name2}\npatient:{ft1:.2f}\treference:{r}\nGR_percentage:{s1 * 100:.2f}")
#         sum2 += s1 * 100
#         green.append(s1 * 100)
#         cv2.line(image, tuple(start_pt), tuple(end_pt), (0, 255, 0), 2)
#         midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
#         text_position = (midpoint[0], midpoint[1] - 10)
#         cv2.putText(image, f"{s1 * 100}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
#     return image, sum2, green, distances  # Return the list of distances


# def apply_filter(image, landmarks, landmark_indices):
#     for idx in landmark_indices:
#         landmark_pt = tuple(map(int, landmarks[idx]))
#         cv2.circle(image, landmark_pt, 3, (0, 255, 0), -1)
#     return image

# def calculate_golden_ratio(input_path, output_path):
#     ratios = []
#     image = cv2.imread(input_path)
#     if image is None:
#         raise FileNotFoundError(f"Failed to load image from: {input_path}")

#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh()

#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             height, width, _ = image.shape
#             landmarks = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark]
#             landmark_indices = [lm['start'] for lm in l1] + [lm['end'] for lm in l1]
#             image = apply_filter(image, landmarks, landmark_indices)
#             landmark_indices1 = [lm['start'] for lm in l2] + [lm['end'] for lm in l2]
#             image1 = apply_filter(image, landmarks, landmark_indices1)
#             image, rsum, redlist, ft_distances = idraw_lines_with_text(image, landmarks, l1)
#             image1, gsum, greenlist, ft1_distances = idraw_lines_with_text1(image1, landmarks, l2)
#             asum = rsum + gsum
#             avg = asum / (len(l1) + len(l2))
#             print(f"percentage: {avg:.2f}%")

#             # Prepare ratio data
#             ft_combined = ft_distances + ft1_distances  # Combine the distances
#             gr_percentage = redlist + greenlist
#             for idx,(name, patient_value, reference_value,g) in enumerate(zip(Name + Name2, ft_combined, rft_arr + rft1_arr,gr_percentage)):

#                 ratios.append({
#                     'Name': name,
#                     'patient_value': f"{patient_value:.3f}",
#                     'reference_value': f"{reference_value:.3f}",
#                     'gr_percentage': f"{g:.3f}"
#                 })


#             # for landmark_pt in landmarks:
#             #     cv2.circle(image, landmark_pt, 1, (255, 0, 0), -1)

#     cv2.imwrite(output_path, image)
#     return output_path, ratios, avg

# inpt_arr=[]
# inpt1_arr=[]
# #value1

# def rdraw_lines_with_text(image, landmarks, landmark_pairs):
#     reference_real_world_size = 3.5

#     for pair in landmark_pairs:
#         start_idx = pair['start']
#         end_idx = pair['end']
#         start_pt = landmarks[start_idx]
#         end_pt = landmarks[end_idx]
#         distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
#         inpt_arr.append(float(f"{distance:.3f}"))

#         cv2.line(image, start_pt, end_pt, (0, 0, 255, 0), 2)

#         # Calculate the midpoint
#         midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

#         # Adjust the position of the text to be above the line
#         text_position = (midpoint[0], midpoint[1] - 10)

#         #Annotate with the label
#         #cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     return image

# # value 1.618

# def rdraw_lines_with_text1(image, landmarks, landmark_pairs):
#     reference_real_world_size = 3.5

#     for pair in landmark_pairs:
#         start_idx = pair['start']
#         end_idx = pair['end']
#         start_pt = landmarks[start_idx]
#         end_pt = landmarks[end_idx]
#         distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
#         inpt1_arr.append(float(f"{distance:.3f}"))

#         #Draw the line
#         cv2.line(image, start_pt, end_pt, (0, 255, 0), 2)

#         # Calculate the midpoint
#         midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

#         # Adjust the position of the text to be above the line
#         text_position = (midpoint[0], midpoint[1] - 10)

#         #Annotate with the label
#        # cv2.putText(image, str(w), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     return image

# def image_per(image_path_var,output_path):

#     ratios1=[]
#     inpt_arr.clear()
#     inpt1_arr.clear()
#     image = cv2.imread(image_path_var)
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh()

#     if image is None:
#        raise FileNotFoundError(f"Failed to load image from: {image_path_var}")


#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# # Process the image with MediaPipe face mesh
#     results = face_mesh.process(image_rgb)
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#           height, width, _ = image.shape
#           l= []
#           landmarks = [(int(landmark.x * width),int(landmark.y * height)) for landmark in face_landmarks.landmark]


#           landmark_indices = [lm['start'] for lm in filter_landmark3] + [lm['end'] for lm in filter_landmark3]
#           image = apply_filter(image, landmarks, landmark_indices)
#           landmark_indices1 = [lm['start'] for lm in filter_landmark4] + [lm['end'] for lm in filter_landmark4]
#           image = apply_filter(image, landmarks, landmark_indices1)

#           image = rdraw_lines_with_text(image, landmarks, filter_landmark3)
#           image= rdraw_lines_with_text1(image, landmarks, filter_landmark4)


#           # for landmark_pt in landmarks:
#           #
#           #     cv2.circle(image,landmark_pt, 1, (255, 0, 0), -1)  # Draw a small dot for each landmark


#     for i, j, r1, r2 in zip(inpt_arr, inpt1_arr, Name, Name2):
#         p = i * 1.618
#         q = p / j
#         if q * 100 > 100:
#             q = j / p
#             q * 100
#             ratios1.append({
#                 'Description1': f"{r1}",
#                 'Description2': f"{r2}",
#                 'dist1': f"{i:.3f}",
#                 'dist2': f"{j:.3f}",
#                 'Percentage': f"{q * 100:.3f}"
#             })
#         else:
#             q * 100
#             ratios1.append({
#                 'Description1': f"{r1}",
#                 'Description2': f"{r2}",
#                 'dist1': f"{i:.3f}",
#                 'dist2': f"{j:.3f}",
#                 'Percentage': f"{q * 100:.3f}"
#             })

#     avg_percentage = sum([float(ratio['Percentage']) for ratio in ratios1]) / len(ratios1)
#     cv2.imwrite(output_path, image)
#     return output_path, ratios1, avg_percentage







