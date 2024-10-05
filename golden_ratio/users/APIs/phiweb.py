import os
import mediapipe as mp
import numpy as np
import cv2
# import re
# import random
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
global_perc = []
details = []
ratios = []


def euclidean_distance(point1, point2):

    reference_real_world_size = 3.5
    x1 = landlist[point1][0]
    x2 = landlist[point2][0]
    y1 = landlist[point1][1]
    y2 = landlist[point2][1]
    xdiff = x2 - x1
    ydiff = y2 - y1

    return np.linalg.norm(np.array(xdiff) - np.array(ydiff)) / reference_real_world_size  # dist

def draw_line(landlist, image, pt1, pt2):
    cv2.line(image, landlist[pt1], landlist[pt2], (0, 0, 255, 0), 2)

def ratio(d1, d2):
    # temp = d1 / d2 # given ratio
    if d1 > d2:
        temp = d1 / d2
    else:
        temp = d2 / d1
    if temp > 1.618:
        ratio_percentage = 1.618 / temp
        ratio_percentage = ratio_percentage*100
    else:
        ratio_percentage = temp / 1.618
        ratio_percentage = ratio_percentage * 100
    return temp, ratio_percentage

def enfnb_1_ratio(image, landlist):
            # 27-29/29-33
            dist1 = euclidean_distance(168, 5)
            draw_line(landlist, image, 168, 5)

            dist2 = euclidean_distance(5, 2)
            draw_line(landlist, image, 5, 2)
            face_ratio, ratio_per = ratio(dist1, dist2)
            global_perc.append(ratio_per)
            details.append(
                f" Eyes to Nose flair\nNose flair to Nose base\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def entcl_2_ratio(image, landlist):

    dist1 = euclidean_distance(168, first_custom_landmark_index9)
    draw_line(landlist, image, 168, first_custom_landmark_index9)

    dist2 = euclidean_distance(first_custom_landmark_index9, 14)
    draw_line(landlist, image, first_custom_landmark_index9, 13)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Eyes to Nostril top\nNostril top to Center of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def enbbl_3_ratio(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(168, 2)
    draw_line(landlist, image, 168, 2)

    dist2 = euclidean_distance(2, 17)
    draw_line(landlist, image, 2, 17)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Eyes to Nose base\nNose base to Bottom of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def eclbc_4_ratio(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(first_custom_landmark_index7, 13)
    draw_line(landlist, image, first_custom_landmark_index7, 13)

    dist2 = euclidean_distance(13, 152)
    draw_line(landlist, image, 13, 152)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Eyes to Center of lips\nCenter of lips to Bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def nfblbc_5_ratio(image, landlist):

    dist1 = euclidean_distance(first_custom_landmark_index8, 17)
    draw_line(landlist, image, first_custom_landmark_index8, 17)
    dist2 = euclidean_distance(17, 152)
    draw_line(landlist, image, 17, 152)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Nose flair to Bottom of lips\nBottom of lips to Bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def nftlbc_6_ratio(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(37, 5)
    tempdist = euclidean_distance(267, 5)
    dist1 = (dist1 + tempdist) / 2
    draw_line(landlist, image, 0, 5)

    dist2 = euclidean_distance(37, 152)
    tempdist2 = euclidean_distance(267, 152)
    dist2 = (dist2 + tempdist2) / 2
    draw_line(landlist, image, 0, 152)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Nose flair to Top of lips\nTop of lips to Bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def tlblbc_7_ratio(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(37, 17)
    tempdist = euclidean_distance(267, 17)
    dist1 = (dist1 + tempdist) / 2
    draw_line(landlist, image, 0, 17)

    dist2 = euclidean_distance(17, 152)
    draw_line(landlist, image, 17, 152)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of lips to Bottom of lips\nBottom of lips to Bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def tlclbl_8_ratio(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(37, 14)
    tempdist = euclidean_distance(267, 14)
    dist1 = (dist1 + tempdist) / 2
    draw_line(landlist, image, 0, 13)

    dist2 = euclidean_distance(14, 17)
    draw_line(landlist, image, 13, 17)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of lips to Center of lips\nCenter of lips to Bottom of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def aetebe_9_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(105, 159)
    tempdist = euclidean_distance(66, 159)
    dist1 = (dist1 + tempdist) / 2
    draw_line(landlist, image, 105, 159)

    dist2 = euclidean_distance(159, 145)
    draw_line(landlist, image, 159, 145)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Arc of eyebrows to Top of eyes\nTop of eyes to Bottom of eyes -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def aetebe_9_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(334, 386)
    tempdist = euclidean_distance(296, 386)
    dist1 = (dist1 + tempdist) / 2
    draw_line(landlist, image, 334, 386)

    dist2 = euclidean_distance(374, 386)
    draw_line(landlist, image, 374, 386)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Arc of eyebrows to Top of eyes\nTop of eyes to Bottom of eyes -> Left_side\nDistance:({dist1:.3f},{dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def aetlbc_10_ratio_right(image, landlist):
    dist1 = euclidean_distance(52, 37)
    tempdist = euclidean_distance(53, 37)
    dist1 = (dist1 + tempdist) / 2
    draw_line(landlist, image, 34, 33)

    dist2 = euclidean_distance(37, 152)
    draw_line(landlist, image, 33, 468)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Arc of eyebrows to Top of lips\nTop of lips to bottom of chin -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def aetlbc_10_ratio_left(image, landlist):
    dist1 = euclidean_distance(282, 267)
    tempdist = euclidean_distance(283, 267)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 334, 267)

    dist2 = euclidean_distance(267, 152)
    draw_line(landlist, image, 267, 152)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Arc of eyebrows to top of lips\nTop of lips to bottom of chin -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfoecp_11_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(34, 33)
    draw_line(landlist, image, 34, 33)

    dist2 = euclidean_distance(33, 468)
    draw_line(landlist, image, 33, 468)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Outside of eyes\nOutside of eyes to Center of pupil -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfoecp_11_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(264, 263)
    draw_line(landlist, image, 264, 263)

    dist2 = euclidean_distance(263, 473)
    draw_line(landlist, image, 263, 473)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Outside of eyes\nOutside of eyes to Center of pupil -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfoiieie_12_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(34, 471)
    draw_line(landlist, image, 34, 471)

    dist2 = euclidean_distance(471, 173)
    draw_line(landlist, image, 471, 173)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Outside of iris\nOutside of iris to Inside of eye and Inside of eyebrow -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfoiieie_12_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(264, 474)
    draw_line(landlist, image, 264, 474)

    dist2 = euclidean_distance(474, 398)
    draw_line(landlist, image, 474, 398)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Outside of iris\nOutside of iris to Inside of eye and Inside of eyebrow -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfiicf_13_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(34, 469)
    draw_line(landlist, image, 34, 469)
    dist2 = euclidean_distance(469, 168)
    draw_line(landlist, image, 469, 168)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Inside of iris\nInside of iris to Center of face -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfiicf_13_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(264, 476)
    draw_line(landlist, image, 264, 476)

    dist2 = euclidean_distance(476, 168)
    draw_line(landlist, image, 476, 168)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Inside of iris\nInside of iris to Center of face -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfineioeie_14_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(34, 243)
    draw_line(landlist, image, 34, 243)

    dist2 = euclidean_distance(243, 362)
    draw_line(landlist, image, 243, 362)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Inside of near eye\nInside of near eye to Inside of opposite eye and Inside of eyebrows -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfineioeie_14_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(264, 463)
    draw_line(landlist, image, 264, 463)

    dist2 = euclidean_distance(463, 133)
    draw_line(landlist, image, 463, 133)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Inside of near eye\nInside of near eye to Inside of opposite eye and Inside of eyebrows -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfcfooe_15_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(34, 168)
    draw_line(landlist, image, 34, 168)

    dist2 = euclidean_distance(168, 263)
    draw_line(landlist, image, 168, 263)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Center of face\nCenter of face to Outside of opposite eye -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfcfooe_15_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(264, 168)
    draw_line(landlist, image, 264, 168)

    dist2 = euclidean_distance(168, 33)
    draw_line(landlist, image, 168, 33)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Center of face\nCenter of face to Outside of opposite eye -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfioeosf_16_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(34, 362)
    draw_line(landlist, image, 34, 362)

    dist2 = euclidean_distance(362, 264)
    draw_line(landlist, image, 362, 264)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Inside opposite eye\nInside opposite eye to  Opposite side of face -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def sfioeosf_16_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(264, 133)
    draw_line(landlist, image, 264, 133)

    dist2 = euclidean_distance(133, 34)
    draw_line(landlist, image, 133, 34)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f"Side of face to Inside opposite eye\nInside opposite eye to  Opposite side of face -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def oefn_17_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(226, 244)
    draw_line(landlist, image, 226, 244)

    dist2 = euclidean_distance(217, 437)
    draw_line(landlist, image, 217, 437)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Outside of eye\nFlair of nose -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def oefn_17_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(446, 464)
    draw_line(landlist, image, 446, 464)

    dist2 = euclidean_distance(437, 217)
    draw_line(landlist, image, 437, 217)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Outside of eye\nFlair of nose  -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def cptnfbn_18_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(473, 330)
    draw_line(landlist, image, 473, 330)

    dist2 = euclidean_distance(420, 305)
    draw_line(landlist, image, 420, 305)

    face_ratio, ratio_per = ratio(dist1, dist2)

    global_perc.append(ratio_per)
    details.append(
        f" Center of pupils to Top of nose flair\nTop of nose flair to Bottom of nose -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def cptnfbn_18_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(468, 101)
    draw_line(landlist, image, 468, 101)

    dist2 = euclidean_distance(198, 75)
    draw_line(landlist, image, 198, 75)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Center of pupils to Top of nose flair\nTop of nose flair to Bottom of nose -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def cpbncn_19_ratio_right(image, landlist):
    # 27-29/29-33
    # dist1 = euclidean_distance(417,363)
    dist1 = euclidean_distance(417, 420)
    tempdist = euclidean_distance(417, 281)
    dist1 = (dist1 + tempdist) / 2
    draw_line(landlist, image, 417, 456)
    dist2 = euclidean_distance(420, 328)
    tempdist = euclidean_distance(281, 328)
    draw_line(landlist, image, 456, 328)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Center of pupils to Bridge of nose\nBridge of nose to Center of nostril -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def cpbncn_19_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(193, 236)
    draw_line(landlist, image, 193, 236)
    tempdist = euclidean_distance(193, 134)
    dist1 = (dist1 + tempdist) / 2
    dist2 = euclidean_distance(236, 99)
    draw_line(landlist, image, 236, 99)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Center of pupils to Bridge of nose\nBridge of nose to Center of nostril -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def iewnin_20_ratio_right(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(414, 417)
    draw_line(landlist, image, 414, 417)

    dist2 = euclidean_distance(141, 370)
    draw_line(landlist, image, 141, 370)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Side of face to Inside of iris\nInside of iris to Center of face -> Right_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def iewnin_20_ratio_left(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(190, 193)
    draw_line(landlist, image, 190, 193)

    dist2 = euclidean_distance(141, 370)
    draw_line(landlist, image, 141, 370)
    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Inside of eyes to Width of nose to Inside of nostril -> Left_side\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def wmcbpp_21_ratio(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(37, 82)
    draw_line(landlist, image, 37, 82)

    dist2 = euclidean_distance(14, 17)
    draw_line(landlist, image, 14, 17)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Width of mouth\nCupid's bow point of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def bntlbl_22_ratio(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(94, 37)
    tempdist = euclidean_distance(94, 267)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 326, 267)
    dist2 = euclidean_distance(267, 17)
    tempdist = euclidean_distance(37, 17)
    dist2 = (tempdist + dist2) / 2
    draw_line(landlist, image, 267, 314)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Bottom of nose to Top of lips\nTop of lips to Bottom of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add1(image, landlist):
    dist1 = euclidean_distance(10, 168)
    tempdist = euclidean_distance(10, 6)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 10, 168)

    dist2 = euclidean_distance(168, 4)
    tempdist = euclidean_distance(6, 4)
    dist2 = (tempdist + dist2) / 2
    draw_line(landlist, image, 168, 4)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to centre of eyes\nCentre of eyes to nosetip\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add2(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(10, 168)
    tempdist = euclidean_distance(10, 6)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 10, 168)

    dist2 = euclidean_distance(14, 152)
    draw_line(landlist, image, 14, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to centre of eyes\nCentre of lips to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add3(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(10, 168)
    tempdist = euclidean_distance(10, 6)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 10, 168)

    dist2 = euclidean_distance(10, 4)
    draw_line(landlist, image, 10, 4)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to centre of eyes\nTop of head to nosetip\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add4(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(10, 4)
    draw_line(landlist, image, 10, 4)

    dist2 = euclidean_distance(168, 14)
    tempdist = euclidean_distance(6, 14)
    dist2 = (tempdist + dist2) / 2
    draw_line(landlist, image, 168, 14)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to nosetip\nCentre of eyes to centre of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add5(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(10, 4)
    draw_line(landlist, image, 10, 4)

    dist2 = euclidean_distance(4, 152)
    draw_line(landlist, image, 4, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to nosetip\nNosetip to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add6(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(10, 4)
    draw_line(landlist, image, 10, 4)

    dist2 = euclidean_distance(10, 152)
    draw_line(landlist, image, 10, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to nosetip\nTop of head to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add7(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(168, 152)
    tempdist = euclidean_distance(6, 152)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 168, 152)

    dist2 = euclidean_distance(10, 152)
    draw_line(landlist, image, 10, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to bottom of chin\nCentre of eyes to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add9(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(168, 4)
    tempdist = euclidean_distance(6, 4)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 168, 4)

    dist2 = euclidean_distance(168, 14)
    tempdist = euclidean_distance(6, 14)
    dist2 = (tempdist + dist2) / 2
    draw_line(landlist, image, 168, 14)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Centre of eyes to nosetip\nCentre of eyes to centre of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add11(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(10, 168)
    tempdist = euclidean_distance(10, 6)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 10, 168)

    dist2 = euclidean_distance(168, 152)
    tempdist = euclidean_distance(6, 152)
    dist2 = (tempdist + dist2) / 2
    draw_line(landlist, image, 168, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Top of head to centre of eyes\nCentre of eyes to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add12(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(168, 14)
    tempdist = euclidean_distance(6, 14)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 10, 168)

    dist2 = euclidean_distance(168, 152)
    tempdist = euclidean_distance(6, 152)
    dist2 = (tempdist + dist2) / 2
    draw_line(landlist, image, 168, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Centre of eyes to centre of lips\nCentre of eyes to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add13(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(168, 152)
    tempdist = euclidean_distance(6, 152)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 168, 152)

    dist2 = euclidean_distance(4, 152)
    draw_line(landlist, image, 4, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Centre of eyes to bottom of chin\nNosetip to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add14(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(168, 4)
    tempdist = euclidean_distance(6, 4)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 168, 4)

    dist2 = euclidean_distance(4, 152)
    draw_line(landlist, image, 4, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Centre of eyes to nosetip\nNosetip to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add15(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(4, 152)
    draw_line(landlist, image, 4, 152)

    dist2 = euclidean_distance(14, 152)
    draw_line(landlist, image, 14, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Nosetip to bottom of chin\nCentre of lips to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add16(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(4, 13)
    tempdist = euclidean_distance(1, 13)
    dist1 = (tempdist + dist1) / 2
    draw_line(landlist, image, 4, 13)

    dist2 = euclidean_distance(13, 152)
    draw_line(landlist, image, 13, 152)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Nosetip to centre of lips\nCentre of lips to bottom of chin\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add17(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(127, 356)
    draw_line(landlist, image, 127, 356)

    dist2 = euclidean_distance(33, 263)
    draw_line(landlist, image, 33, 263)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Width of head\nDistance between outer edge of two eyes\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add18(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(33, 263)
    draw_line(landlist, image, 33, 263)

    dist2 = euclidean_distance(61, 291)
    draw_line(landlist, image, 61, 291)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Distance between outer edge of two eyes\nWidth of lips\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def add19(image, landlist):
    # 27-29/29-33
    dist1 = euclidean_distance(61, 291)
    draw_line(landlist, image, 61, 291)

    dist2 = euclidean_distance(219, 439)
    draw_line(landlist, image, 219, 439)

    face_ratio, ratio_per = ratio(dist1, dist2)
    global_perc.append(ratio_per)
    details.append(
        f" Width of lips\nWidth of nose\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def face_length(image, landlist):
    dist1 = euclidean_distance(first_custom_landmark_index, 152)
    draw_line(landlist, image, first_custom_landmark_index, 152)
    dist2 = euclidean_distance(227, 447)
    draw_line(landlist, image, 227, 447)
    face_ratio, ratio_per = ratio(dist1, dist2)
    details.append(
        f"#Face Ratio\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def forehead_ratio(image, landlist):
    dist1 = euclidean_distance(first_custom_landmark_index, first_custom_landmark_index1)
    draw_line(landlist, image, first_custom_landmark_index, first_custom_landmark_index1)
    dist2 = euclidean_distance(103, 332)
    draw_line(landlist, image, 103, 332)
    face_ratio, ratio_per = ratio(dist1, dist2)
    details.append(
        f"#Forehead ratio\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def nose2_ratio(image, landlist):
    dist1 = euclidean_distance(first_custom_landmark_index1, 2)
    draw_line(landlist, image, first_custom_landmark_index1, 2)
    dist2 = euclidean_distance(48, 278)
    draw_line(landlist, image, 48, 278)
    face_ratio, ratio_per = ratio(dist1, dist2)
    details.append(
        f"#Nose ratio\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def eyes_ratio(image, landlist):
    dist1 = euclidean_distance(133, 33)
    draw_line(landlist, image, 133, 33)
    dist2 = euclidean_distance(133, first_custom_landmark_index3)
    draw_line(landlist, image, 133, first_custom_landmark_index3)
    face_ratio, ratio_per = ratio(dist1, dist2)
    details.append(
        f"#Eyes ratio\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def eyebrow_ratio(image, landlist):
    dist1 = euclidean_distance(336, 334)
    draw_line(landlist, image, 336, 334)
    dist2 = euclidean_distance(336, first_custom_landmark_index2)
    draw_line(landlist, image, 336, first_custom_landmark_index2)
    face_ratio, ratio_per = ratio(dist1, dist2)
    details.append(
        f"#Eyebrow ratio\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def lips_ratio(image, landlist):
    dist1 = euclidean_distance(0, 13)
    # tempdist=euclidean_distance(267,13)
    # dist1=(tempdist+dist1)/2
    draw_line(landlist, image, 0, 13)
    dist2 = euclidean_distance(13, 17)
    draw_line(landlist, image, 13, 17)
    face_ratio, ratio_per = ratio(dist1, dist2)
    details.append(
        f"#Lips ratio\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def chin_ratio(image, landlist):
    dist1 = euclidean_distance(190, 414)
    draw_line(landlist, image, 190, 414)
    dist2 = euclidean_distance(148, 377)
    draw_line(landlist, image, 148, 377)
    face_ratio, ratio_per = ratio(dist1, dist2)
    details.append(
        f"#Chin ratio\nDistance:({dist1:.3f}, {dist2:.3f}), Percentage: {ratio_per:.3f}\n------------------------------------------------------------------------")

def fun(image,landlist):
    face_length(image, landlist)
    forehead_ratio(image, landlist)
    nose2_ratio(image, landlist)
    eyes_ratio(image, landlist)
    lips_ratio(image, landlist)
    chin_ratio(image, landlist)
    eyebrow_ratio(image, landlist)
    enfnb_1_ratio(image, landlist)  # 0
    entcl_2_ratio(image, landlist)  # 1
    enbbl_3_ratio(image, landlist)  # 2
    eclbc_4_ratio(image, landlist)  # 3
    nfblbc_5_ratio(image, landlist)  # 4
    nftlbc_6_ratio(image, landlist)  # 5
    tlblbc_7_ratio(image, landlist)  # 6
    tlclbl_8_ratio(image, landlist)  # 7
    aetebe_9_ratio_right(image, landlist)  # 8
    aetebe_9_ratio_left(image, landlist)  # 9
    aetlbc_10_ratio_right(image, landlist)
    aetlbc_10_ratio_left(image, landlist)
    sfoecp_11_ratio_right(image, landlist)  # 10
    sfoecp_11_ratio_left(image, landlist)  # 11
    sfoiieie_12_ratio_right(image, landlist)  # 12
    sfoiieie_12_ratio_left(image, landlist)  # 13
    sfiicf_13_ratio_right(image, landlist)  # 14
    sfiicf_13_ratio_left(image, landlist)  # 15
    sfineioeie_14_ratio_right(image, landlist)  # 16
    sfineioeie_14_ratio_left(image, landlist)  # 17
    sfcfooe_15_ratio_right(image, landlist)  # 18
    sfcfooe_15_ratio_left(image, landlist)  # 19
    sfioeosf_16_ratio_right(image, landlist)  # 20
    sfioeosf_16_ratio_left(image, landlist)  # 21
    # oefn_17_ratio_right(image,landlist)#22
    oefn_17_ratio_left(image, landlist)  # 23
    cptnfbn_18_ratio_right(image, landlist)  # 24
    cptnfbn_18_ratio_left(image, landlist)  # 25
    # cpbncn_19_ratio_right(image,landlist)#26
    cpbncn_19_ratio_left(image, landlist)  # 27
    iewnin_20_ratio_right(image, landlist)  # 28
    iewnin_20_ratio_left(image, landlist)  # 29
    wmcbpp_21_ratio(image, landlist)  # 30
    bntlbl_22_ratio(image, landlist)
    add1(image, landlist)
    add2(image, landlist)
    add3(image, landlist)
    add4(image, landlist)
    add5(image, landlist)
    add6(image, landlist)
    add7(image, landlist)
    add9(image, landlist)
    add11(image, landlist)
    add12(image, landlist)
    add13(image, landlist)
    add14(image, landlist)
    add15(image, landlist)
    add16(image, landlist)
    add17(image, landlist)
    add18(image, landlist)
    add19(image, landlist)
     
image_path='C:/Users/user/PycharmProjects/goldebn ratio/PycharmProjects/pythonProject/app_folder/input_image/f.jpg'
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
        l = []
        landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in
                            face_landmarks.landmark]
        # print(len(landmark_points))
        mid_forehead_x, mid_forehead_y = landmark_points[10]
        hairline_landmarks = [
            (mid_forehead_x, mid_forehead_y - 50),  # Point directly above middle forehead
            (mid_forehead_x - 30, mid_forehead_y - 40),  # Point slightly left
            (mid_forehead_x + 30, mid_forehead_y - 40),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[8]
        eye_landmarks = [
            (mid_forehead_x, mid_forehead_y + 13),  # Point directly above middle forehead
            (mid_forehead_x, mid_forehead_y + 30),  # Point slightly left
            (mid_forehead_x + 30, mid_forehead_y - 40),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[301]
        brow_landmarks = [
            (mid_forehead_x, mid_forehead_y - 3),  # Point directly above middle forehead
            (mid_forehead_x - 30, mid_forehead_y - 40),  # Point slightly left
            (mid_forehead_x + 30, mid_forehead_y - 40),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[34]
        side_landmarks = [
            (mid_forehead_x, mid_forehead_y - 8),  # Point directly above middle forehead
            (mid_forehead_x - 30, mid_forehead_y - 40),  # Point slightly left
            (mid_forehead_x + 30, mid_forehead_y - 40),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[0]
        lip_landmarks = [
            (mid_forehead_x, mid_forehead_y - 3),  # Point directly above middle forehead
            (mid_forehead_x - 30, mid_forehead_y - 40),  # Point slightly left
            (mid_forehead_x + 30, mid_forehead_y - 40),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[176]
        chin1_landmarks = [
            (mid_forehead_x - 10, mid_forehead_y - 20),  # Point slightly left
            (mid_forehead_x + 30, mid_forehead_y - 40),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[400]
        chin2_landmarks = [
            # Point slightly left
            (mid_forehead_x + 10, mid_forehead_y - 20),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[5]
        nose_landmarks = [
            # Point slightly left
            (mid_forehead_x, mid_forehead_y + 10),  # Point slightly right
        ]
        mid_forehead_x, mid_forehead_y = landmark_points[1]
        nostril_landmarks = [
            # Point slightly left
            (mid_forehead_x, mid_forehead_y - 10),  # Point slightly right
        ]
        # Combine with original landmarks for visualization
        landlist = landmark_points + hairline_landmarks + eye_landmarks + brow_landmarks + side_landmarks + lip_landmarks + chin1_landmarks + chin2_landmarks + nose_landmarks + nostril_landmarks
        first_custom_landmark_index = len(landmark_points)
        first_custom_landmark_index1 = len(landmark_points + hairline_landmarks)
        first_custom_landmark_index7 = len(landmark_points + hairline_landmarks) + 1
        first_custom_landmark_index2 = len(landmark_points + hairline_landmarks + eye_landmarks)
        first_custom_landmark_index3 = len(landmark_points + hairline_landmarks + eye_landmarks + brow_landmarks)
        first_custom_landmark_index4 = len(
            landmark_points + hairline_landmarks + eye_landmarks + brow_landmarks + side_landmarks)
        first_custom_landmark_index5 = len(
            landmark_points + hairline_landmarks + eye_landmarks + brow_landmarks + side_landmarks + lip_landmarks)
        first_custom_landmark_index6 = len(
            landmark_points + hairline_landmarks + eye_landmarks + brow_landmarks + side_landmarks + lip_landmarks + chin1_landmarks)
        first_custom_landmark_index8 = len(
            landmark_points + hairline_landmarks + eye_landmarks + brow_landmarks + side_landmarks + lip_landmarks + chin1_landmarks + chin2_landmarks)
        first_custom_landmark_index9 = len(
            landmark_points + hairline_landmarks + eye_landmarks + brow_landmarks + side_landmarks + lip_landmarks + chin1_landmarks + chin2_landmarks + nose_landmarks)

        # for landmark_pt in landlist:
        #     cv2.circle(image, tuple(map(int, landmark_pt)), 1, (255, 0, 0), -1)  # Draw a small dot for each landmark

fun(image,landlist)       
avg = sum(global_perc) / len(global_perc)*100 if global_perc else 0
avg = avg*0.01
print(avg)
cv2.imshow("Filtered_Image", image)
# print(image)
cv2.waitKey(0)
cv2.destroyAllWindows()

