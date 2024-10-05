import os
import mediapipe as mp
import numpy as np
import cv2
index=[151,195,18,143,265,468,473,6,54,67,10,297,284,137,205,45,423,352,58,212,0,410,367]
index1=[151,195,18,143,265,468,473,6]
index2=[54,67,10,297,284,137,205,45,423,352,58,212,0,410,367]
sym_asym_name = [
        'Hairline to centre of eyebrows',
        'Centre of eyebrows to bottom of nose',
        'Bottom of nose to bottom of chin'
    ]
sym_asym_name2 = [
         'Right side to edge of inner eye',
         'Left side to edge of inner eye',
         'Right inner eye',
         'Left inner eye',
         'Between eyes'
        ]
#===================================================================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
list1 = []
list2 = []
list3 = []
list4 = []
details = []
details1 = []
a1 = []
a2 = []
a3 = []
ans = []
def draw_vertical_line(image, x):
    ih, iw, _ = image.shape
    # Draw the vertical line
    cv2.line(image, (x, 0), (x, ih), (0, 255, 0), 2)


def draw_horizontal_line(image, y):
    ih, iw, _ = image.shape
    # Draw the horizontal line
    cv2.line(image, (0, y), (iw, y), (0, 0, 255), 2)

    #===================================================================================================
def euclidean_distance(point1, point2):
    reference_real_world_size = 3.5
    x1 = landlist[point1][0]
    x2 = landlist[point2][0]
    y1 = landlist[point1][1]
    y2 = landlist[point2][1]
    xdiff = x2 - x1
    ydiff = y2 - y1
    # distsq = pow(xdiff,2) + pow(ydiff,2)
    # dist = math.sqrt(distsq)

    return np.linalg.norm(np.array(xdiff) - np.array(ydiff)) / reference_real_world_size  # dist


def draw_line(landlist, image, pt1, pt2):
    #     cv2.line(image, landlist[pt1], landlist[pt2], (0, 0, 255, 0), 2)
    pass



def overall_face(image, landlist):
    # 1
    dist1 = euclidean_distance(first_custom_landmark_index, 152)
    draw_line(landlist, image, first_custom_landmark_index, 152)
    #details.append('overall_face\nDistance:',dist1)
    details.append(('Overall face(Reference)',dist1))
    list1.append(dist1)



def hairline_to_centre_of_eyebrows(image, landlist):
    # 2
    dist1 = euclidean_distance(first_custom_landmark_index, 9)
    draw_line(landlist, image, first_custom_landmark_index, 9)
    details.append(('Hairline to centre of eyebrows',dist1))
    # print(dist1)
    list2.append(dist1)



def centre_of_eyebrows_to_bottom_of_nose(image, landlist):
    # 3
    dist1 = euclidean_distance(9, 94)
    draw_line(landlist, image, 9, 94)
    details.append(('Centre of eyebrows to bottom of nose',dist1))
    # print(dist1)
    list2.append(dist1)



def bottom_of_nose_to_bottom_of_chin(image, landlist):
    # 4
    dist1 = euclidean_distance(94, 152)
    draw_line(landlist, image, 94, 152)
    details.append(('Bottom of nose to bottom of chin',dist1))
    # print(dist1)
    list2.append(dist1)



# ================
def upper_cheek(image, landlist):
    dist1 = euclidean_distance(234, 454)
    draw_line(landlist, image, 234, 454)
    details.append(('Upper cheeks(Reference)',dist1))
    # print(dist1)
    list3.append(dist1)


def right_ear_to_edge_of_inner_eye(image, landlist):
    dist1 = euclidean_distance(33, 127)
    draw_line(landlist, image, 33, 127)
    details.append(('Right side to edge of inner eye',dist1))
    # print(dist1)
    list4.append(dist1)


def left_ear_to_edge_of_inner_eye(image, landlist):
    dist1 = euclidean_distance(263, 356)
    draw_line(landlist, image, 263, 356)
    details.append(('Left side to edge of inner eye',dist1))
    # print(dist1)
    list4.append(dist1)


def right_inner_eye(image, landlist):
    dist1 = euclidean_distance(33, 133)
    draw_line(landlist, image, 33, 133)
    details.append(('Right inner eye',dist1))
    # print(dist1)
    list4.append(dist1)


def left_inner_eye(image, landlist):
    dist1 = euclidean_distance(362, 263)
    draw_line(landlist, image, 362, 263)
    details.append(('Left inner eye',dist1))
    # print(dist1)
    list4.append(dist1)



def between_eye(image, landlist):
    dist1 = euclidean_distance(133, 362)
    draw_line(landlist, image, 133, 362)
    details.append(('Between eyes',dist1))
    # print(dist1)
    list4.append(dist1)


def combine():
    overall_face(image, landlist),
    hairline_to_centre_of_eyebrows(image, landlist),
    centre_of_eyebrows_to_bottom_of_nose(image, landlist),
    bottom_of_nose_to_bottom_of_chin(image, landlist),

    upper_cheek(image, landlist),
    right_ear_to_edge_of_inner_eye(image, landlist),
    left_ear_to_edge_of_inner_eye(image, landlist),
    right_inner_eye(image, landlist), left_inner_eye(image, landlist),
    between_eye(image, landlist)

def line(image):
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        results = face_mesh.process(image_rgb)

        # Draw lines on specific landmarks if face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract all detected landmarks
                height, width, _ = image.shape
                landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in
                                   face_landmarks.landmark]

                # Custom landmark indices
                hairline_landmarks = [(landmark_points[10][0], landmark_points[10][1] - 50)]
                eyebrow_landmarks = [(landmark_points[9][0], landmark_points[9][1])]
                nosebase_landmarks = [(landmark_points[94][0], landmark_points[94][1])]
                chin_landmarks = [(landmark_points[152][0], landmark_points[152][1])]

                # Draw vertical lines
                draw_vertical_line(image, landmark_points[127][0])  # Right face edge
                draw_vertical_line(image, landmark_points[33][0])  # Right eye outer corner
                draw_vertical_line(image, landmark_points[133][0])  # Right eye inner corner
                draw_vertical_line(image, landmark_points[362][0])  # Left eye inner corner
                draw_vertical_line(image, landmark_points[263][0])  # Left eye outer corner
                draw_vertical_line(image, landmark_points[356][0])  # Left face edge

                # Draw horizontal lines
                draw_horizontal_line(image, hairline_landmarks[0][1])
                draw_horizontal_line(image, eyebrow_landmarks[0][1])
                draw_horizontal_line(image, nosebase_landmarks[0][1])
                draw_horizontal_line(image, chin_landmarks[0][1])
    return image


def cal():
    for v in list1:
        v1 = v / 3
        # print(v1)
        # print(v)
    for v2, n in zip(list2, sym_asym_name):
        # print(v2)
        if v2 > v1:
            temp = (v1 / v2) * 100
            # text.append(temp)
        else:
            temp = (v2 / v1) * 100
            # text.append(temp)

        if v2 == v1:
            #print(f"{n} - perfect - {temp}")
            d1=(n,"perfect",temp)
            details1.append(d1)
        elif v2 < v1:
            #print(f"{n} - smallest - {temp}")
            d1=(n,"smaller",temp)
            details1.append(d1)
        elif v2 > v1:
            #print(f"{n} - larger - {temp}")
            d1=(n,"larger",temp)
            details1.append(d1)
        ans.append(temp)

def cal1():
    for v in list3:
        v1 = v / 5#
        # print(v1)
    for v2, n in zip(list4, sym_asym_name2):
        # print(v2)
        if v2 > v1:
            temp = (v1 / v2) * 100
        else:
            temp = (v2 / v1) * 100



        if v2 == v1:
            #print(f"{n} - perfect - {temp}")
            d1=(n,"perfect",temp)
            details1.append(d1)
        elif v2 < v1:
            #print(f"{n} - smallest - {temp}")
            d1=(n,"smaller",temp)
            details1.append(d1)
        elif v2 > v1:
            #print(f"{n} - larger - {temp}")
            d1=(n,"larger",temp)
            details1.append(d1)
        ans.append(temp)

def res():

    # print(ans)
    vertic=ans[0:3]
    horizon=ans[3:8]
    avg1 = (vertic[0] + horizon) / 2
    avg2 = (vertic[1] + horizon) / 2
    avg3 = (vertic[2] + horizon) / 2
    #print(avg3)
    for i in horizon:
        avg1 = (vertic[0] + i) / 2
        avg2 = (vertic[1] + i) / 2
        avg3= (vertic[2]+i)/2
        a1.append(avg1)
        a2.append(avg2)
        a3.append(avg3)
    per=ans+a1+a2+a3
    return per

def values(index, per,landlist,image):
    for idx, landmark_pt in enumerate(landlist):
        #cv2.circle(image, tuple(map(int, landmark_pt)), 1, (255, 0, 0), -1)

        for ind, an in zip(index,per):
            if idx == ind:
                cv2.putText(image, f'{an:.2f}', (landmark_pt[0], landmark_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image
image_path='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/app_folder/input_image/im-6.jpg'
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
        landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in
                           face_landmarks.landmark]
        # print(len(landmark_points))
        mid_forehead_x, mid_forehead_y = landmark_points[10]
        hairline_landmarks = [
            (mid_forehead_x, mid_forehead_y - 50),  # Point directly above middle forehead

        ]
        mid_forehead_x, mid_forehead_y = landmark_points[33]
        ear_landmarks = [
            (mid_forehead_x - 5, mid_forehead_y - 4),

        ]
        mid_forehead_x, mid_forehead_y = landmark_points[263]
        ear_landmarksl = [
            (mid_forehead_x + 80, mid_forehead_y - 4),

        ]

        landlist = landmark_points + hairline_landmarks + ear_landmarks + ear_landmarksl
        first_custom_landmark_index = len(landmark_points)
        first_custom_landmark_index1 = len(landmark_points + hairline_landmarks)
        first_custom_landmark_index2 = len(landmark_points + hairline_landmarks + ear_landmarks)
        # for landmark_pt in landlist:
        #     cv2.circle(image, tuple(map(int, landmark_pt)), 1, (255, 0, 0),-1)  # Draw a small dot for each landmark

combine()
cal()
cal1()
per = res()
combined_results = []

symmetry_dict = {item[0]: (item[1], item[2]) for item in details1}


for feature, distance in details:
    if feature in symmetry_dict:
        symmetric, percentage = symmetry_dict[feature]
        combined_results.append([feature, distance, symmetric, percentage])
    else:
        combined_results.append([feature, distance])

ratios = []
for result in combined_results:
    ratio = {
        "Name": result[0],
        "Distance": result[1],
        "Symmetry": result[2] if len(result) > 2 else None,
        "Percentage": result[3] if len(result) > 2 else None
    }
    print(ratio)
    ratios.append(ratio)
#print(ratios)

image_copy1 = image.copy()
image_copy2 = image.copy()
image_copy3 = image.copy()

img11 = values(index1, ans, landlist, image_copy1)
img11 = line(img11)

img12 = values(index2, per, landlist, image_copy2)

img13 = values(index2, per, landlist, image_copy3)
img13 = line(img13)



cv2.imshow('output_path,'  ,img11)
cv2.imshow('output_path1', img12)
cv2.imshow('output_path2', img13)

cv2.waitKey(0)
cv2.destroyAllWindows()


