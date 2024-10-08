from django.shortcuts import render
from .serializers import fileUploadSerializers
from django.views import View
from rest_framework.viewsets import ModelViewSet
from rest_framework.renderers import JSONRenderer
from users.models import UploadedImage
from django.http import HttpResponse, JsonResponse
from rest_framework.response import Response

import os
import json
import mediapipe as mp
import numpy as np


from .landmarks import l1
# from .m2r import filter_landmark3 as filter_landmark3,filter_landmark4 as filter_landmark4
# from .m2r import rft_arr as inpt, rft1_arr as ref
from .m2landmark import l1 as filter_landmark3
from .m2rlandmarks import l2 as filter_landmark4

from .refer import l2, rft_arr, rft1_arr

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle



import re
# from .refer import filter_landmark3 ,filter_landmark4,rft_arr,rft1_arr
# from PIL import Image
import cv2
folder_path = 'golden_ratio_output/'
folder_path1= 'input_as_reference_output/'
folder_path2= 'phi_matrix_output/'
folder_path3= 'image_proportion_sym/'
folder_path4='image_unified_sym/'
folder_path5='image_unified_line_sym/'

# r1=rft_arr
# r2=rft1_arr

# # Define landmark filter with two pairs
# filter_landmark1 = filter_landmark3
# filter_landmark2=filter_landmark4
# # Define paths and filenames


# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()
# ft_arr=[]
# ft1_arr=[]

    
# Name=['Distance between the eyes',
# 'Width of right eye ',
# 'Width of left eye ',
# 'End of the arc to length of the right eyebrow',
# 'End of the arc to length of the left eyebrow',
# 'Right mouth edge to side of the face ',
# 'Left mouth edge to side of the face',
# 'Center of mouth to chin',
# 'Width of upper lip',
# 'Width of the nose',
# 'Width of forehead',
# 'Width of the chin']

class fileUploadView(ModelViewSet):
    queryset=UploadedImage.objects.all()
    serializer_class = fileUploadSerializers


Name = ['Eye Spacing', 'Right eye width', 'Left eye width',
        'Right start of brow to arc', 'Left start of brow to arc',
        'Right oral corner to face side', 'Left oral corner to face side',
        'Oral center to chin', 'Upper lip width', 'Nose width', 'Forehead width',
        'Chin width']

Name2 = ['Right eye corner to face edge', 'Left eye corner to face edge',
         'Right eyebrow width', 'Left eyebrow width', 'Nose length',
         'Oral width', 'Nose tip to oral center', 'Lower lip width',
         'Right eye corner to cheekbone', 'Left eye corner to cheekbone',
         'Middle forehead to right face edge', 'Middle forehead to left face edge']



def idraw_lines_with_text(image, landmarks, landmark_pairs):
    sum1 = 0
    red = []
    #print("input",image)
    reference_real_world_size = 3.5
    distances = []  # List to store distances (ft)
    for pair, n, name in zip(landmark_pairs, rft_arr, Name):
        start_idx = pair['start']
        end_idx = pair['end']
        reference = n #n is aso used for reference both pair['refval'] and n for same list  rft_arr
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt)) / reference_real_world_size
        ft = float(distance)  # Store the distance as float
        distances.append(ft)  # Append the distance to the list
        v = ft / reference
        s1=v/1.618
        if v * 100 > 100:
            s1=1.618/v
            v = reference / ft
            print(f"{name}\npatient:{ft:.2f}\treference:{n}\nGR_percentage:{s1 * 100:.2f}")
        else:
            print(f"{name}\npatient:{ft:.2f}\treference:{n}\nGR_percentage:{s1 * 100:.2f}")
        sum1 += v * 100
        red.append(v * 100)
        # cv2.imread(image)
        cv2.line(image, start_pt, end_pt, (0, 0, 255, 0), 2)
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
        text_position = (midpoint[0], midpoint[1] - 10)
        cv2.putText(image, f"{v * 100}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return image, sum1, red, distances  # Return the list of distances


def idraw_lines_with_text1(image, landmarks, landmark_pairs):
  
    sum2 = 0
    green = []
    reference_real_world_size = 3.5
    distances = []  # List to store distances (ft1)
    for pair, r, name2 in zip(landmark_pairs, rft1_arr, Name2):
        start_idx = pair['start']
        end_idx = pair['end']
        reference = r #r is also used for 'reference' both pair['refval'] and n comes from same list  rft1_arr
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt)) / reference_real_world_size
        ft1 = float(distance)  # Store the distance as float
        distances.append(ft1)  # Append the distance to the list
        c = reference / 1.618
        s = ft1 / c
        s1 = s / 1.618
        if s1 * 100 > 100:
            s1 = 1.618 / s
            print(f"{name2}\npatient:{ft1:.2f}\treference:{r}\nGR_percentage:{s1 * 100:.2f}")
        else:
            print(f"{name2}\npatient:{ft1:.2f}\treference:{r}\nGR_percentage:{s1 * 100:.2f}")
        sum2 += s1 * 100
        green.append(s1 * 100)
        # cv2.imread(image)
        cv2.line(image, start_pt, end_pt, (0, 255, 0), 2)
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
        text_position = (midpoint[0], midpoint[1] - 10)
        cv2.putText(image, f"{s1 * 100}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1) 
    return image, sum2, green, distances  # Return the list of distances
#-----------------------------input as referebce----------------
a=[]
# input_as_reference_Name=['Distance between the eyes',
# 'Width of right eye ',
# 'Width of left eye ',
# 'End of the arc to length of the right eyebrow',
# 'End of the arc to length of the left eyebrow',
# 'Right mouth edge to side of the face ',
# 'Left mouth edge to side of the face',
# 'Center of mouth to chin',
# 'Width of upper lip',
# 'Width of the nose',
# 'Width of forehead',
# 'Width of the chin']

# input_as_reference_Name2=['Distance from right eye inner edge to side of face',
# 'Distance from left eye inner edge to side of face',
# 'Width of the right eyebrow',
# 'Width of the left eyebrow',
# 'Length of the nose',
# 'Width of the mouth',
# 'Starting of the nose to center of the mouth',
# 'Width of lower lip',
# 'Right eye inner edge to cheekbone',
# 'Left eye inner edge to cheekbone',
# 'Centre of forehead to right side of face',
# 'Centre of forehead to left side of face' ]

# inpt_arr=[]
# inpt1_arr=[]
#value1

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


def apply_filter(image, landmarks, landmark_indices):

    for idx in landmark_indices:
        landmark_pt = tuple(map(int, landmarks[idx]))
        cv2.circle(image, landmark_pt, 3, (0, 255, 0), -1)
    return image



class golden_ratio(View):
    def get(self, request, ids, *args, **kwargs):
        input_path = r'/input_image/'
        try:
            filepath = UploadedImage.objects.get(id=ids)
            print("image->",filepath)
        except UploadedImage.DoesNotExist:
            return HttpResponse('Image not found', status=404)
        if 'input_image' in filepath.file_upload.name:
            image_path = filepath.file_upload.path 
            print("file->",image_path)
        else:
            image_path = os.path.join(input_path, filepath.file_upload.name)
        if not os.path.exists(image_path):
            return HttpResponse(f"File not found at path: {image_path}", status=404)
        image = cv2.imread(image_path)
        #print(image)
        if image is None:
            return HttpResponse('Could not read image', status=400)
        _, img_encoded = cv2.imencode('.jpg', image)
        # img_bytes = img_encoded.tobytes()
        
        ratios = [[],[],[]]
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #print(results)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                height, width, _ = image.shape
                landmarks = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark]
                landmark_indices = [lm['start'] for lm in l1] + [lm['end'] for lm in l1]
                image = apply_filter(image, landmarks, landmark_indices)
                landmark_indices1 = [lm['start'] for lm in l2] + [lm['end'] for lm in l2]
                print("as->",landmark_indices1)
                image1 = apply_filter(image, landmarks, landmark_indices1)
                image, rsum, redlist, ft_distances = idraw_lines_with_text(image, landmarks, l1)
                image1, gsum, greenlist, ft1_distances = idraw_lines_with_text1(image1, landmarks, l2)
                asum = rsum + gsum
                avg = asum / (len(l1) + len(l2))
                avg= f"{avg:.3f}"
                print(f"\noverall percentage: {avg}%")


                ft_combined = ft_distances + ft1_distances  # Combine the distances
                gr_percentage = redlist + greenlist
                for idx,(name, patient_value, reference_value,g) in enumerate(zip(Name + Name2, ft_combined, rft_arr + rft1_arr,gr_percentage)):

                    ratios[0].append({
                        'Name': name,
                        'patient_value': f"{patient_value:.3f}",
                        'reference_value': f"{reference_value:.3f}",
                        'gr_percentage': f"{g:.3f}"
                    })
                ratios[1].append({"average":avg})
         

               
                # for landmark_pt in landmarks:
                #     cv2.circle(image, landmark_pt, 1, (255, 0, 0), -1)

        # cv2.imwrite(output_path, image)
        # return  output_path , ratios, avg
        output_path = os.path.join(folder_path,ids+'.jpg')
        cv2.imwrite(output_path, image)
        filepath = UploadedImage.objects.get(id=ids)
        filepath.reference = output_path
        ratios[2].append({
                            'output_image': output_path,
                            
                        })
        print(ratios)
                       

        gr_json='GR_json/'
        json_path = os.path.join(gr_json,f'gr{ids}.json')
        print("JSON->",json_path)
        filepath.reference_json= json_path
        print(output_path)
        filepath.save()    
        # Display the modified image
        # cv2.imshow("golden_ratio", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        
        with open (json_path,'w',encoding="utf-16") as f:
            
                # return f.write(res_json) 
            print("fcre",ratios) 
            json.dump(ratios,f)
        response = Response(ratios, content_type='application/json')
        response.accepted_renderer = JSONRenderer()
        response.accepted_media_type = 'application/json'
        response.renderer_context = {}

        return HttpResponse("response")
        
                # s=json_data()
                # print(type(s))
        
        # return HttpResponse("image succxessfully stored for golden ratio")
        #return HttpResponse()

class GR_retrive(View):
    def get(self, request, ids, *args, **kwargs):
        try:
            # Retrieve the file path from the database using the provided id
            filepath = UploadedImage.objects.get(id=ids)
            reference_json = filepath.reference_json
            
            # Construct the full file path
            file_path = os.path.join(reference_json)
            
            # Open and load the JSON file
            with open(file_path, 'r', encoding="utf-16") as jsonfile:
                json_data = json.load(jsonfile)
            
            # Return the JSON data in the response
            response = Response(json_data, content_type='application/json')
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = 'application/json'
            response.renderer_context = {}
            
            return response
        
        except UploadedImage.DoesNotExist:
            return JsonResponse({"error": "Image not found"}, status=404)
        except FileNotFoundError:
            return JsonResponse({"error": "File not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
        

class GR_pdf(View):
    def get(self, request, ids, *args, **kwargs):

        # Retrieve the file path from the database using the provided id
        filepath = UploadedImage.objects.get(id=ids)
        reference_json = filepath.reference_json
        
        # Construct the full file path
        file_path = os.path.join(reference_json)


        
        # Open and load the JSON file
        with open(file_path, 'r', encoding="utf-16") as jsonfile:
            json_data = json.load(jsonfile)
            
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Define styles
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Golden Ratio Report", styles['Title']))
        elements.append(Spacer(1, 8))

        image_paths = []
        avg_paths = []
        # custom_style = styles.add(ParagraphStyle(name='CustomStyle', fontSize=12, textColor=colors.red))
# Check if json_data is a list or tuple and has more than 2 items
        if isinstance(json_data, (tuple, list)) and len(json_data) >= 3:
            try:
                    # Extract image paths from the nested dictionaries
                    image_paths = [
                        json_data[2][0].get('output_image')
                    ]

                    print("Image paths:", image_paths)  # Printing extracted image paths
            except (KeyError, IndexError) as e:
                    print("Error accessing image paths:", e)

    # Process each image path 



        for i,img_path in enumerate(image_paths):
            if img_path:
                elements.append(Image(img_path, width=440, height=563))

                if i == 0: 
                     elements.append(Spacer(1, 12)) 
                     elements.append(Paragraph("Ploted Coodinates of Golden Ratio", styles['Title']))
                     
               

        if isinstance(json_data, (tuple, list)) and len(json_data) >= 3:
            try:
                    # Extract image paths from the nested dictionaries
                    avg_paths = [
                        json_data[1][0].get('average')
                        
                    ]
                    print("avg paths:", avg_paths)  # Printing extracted image paths
            except (KeyError, IndexError) as e:
                    print("Error accessing image paths:", e)
                      
            for i,avg_path in enumerate(avg_paths):
             if avg_path:
                  elements.append(Paragraph(f"Average Percentage: {avg_path}%", styles['Title']))
                  elements.append(Spacer(1, 24))

        # Handle table data safely
        data_for_table = [['S.No.', 'Name', 'Input\n Distance', 'Reference\n Distance', 'Percentage']]
        # if len(json_data) > 3 and isinstance(json_data[0], list):
        for i, ratio in enumerate(json_data[0]):
                if isinstance(ratio, dict):
                     data_for_table.append([
                        i + 1,
                        ratio.get('Name', 'N/A'),
                        ratio.get('patient_value', 'N/A'),
                        ratio.get('reference_value', 'N/A'),
                        f"{ratio.get('gr_percentage', 'N/A')}%"
                    ])
                   



                    
                    
            # print("DGSGSSUK")
        # Define column widths
        col_widths = [doc.width / 14, doc.width / 1.3, doc.width / 5, doc.width / 9]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)
        d=doc.build(elements)
        buffer.seek(0)
        print(d)
        filepath = UploadedImage.objects.get(id=ids)
        gr_json = 'pdf_gr/'
        pdf_path = os.path.join(gr_json, f'gr{ids}.pdf')
        filepath.reference_pdf= pdf_path
        filepath.save() 

        with open(pdf_path, 'wb') as f:
            f.write(buffer.getvalue())

        # Return PDF as HTTP response
        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="phi{ids}.pdf"'
        return response
    
        # return HttpResponse(response)
        




class input_as_reference(View):

    def get(self, request, ids, *args, **kwargs):
        input_path = r'/input_image/'
        try:
            filepath = UploadedImage.objects.get(id=ids)
            print("image->",filepath)
        except UploadedImage.DoesNotExist:
            return HttpResponse('Image not found', status=404)
        if 'input_image' in filepath.file_upload.name:
            image_path = filepath.file_upload.path 
            print("file->",image_path)
        else:
            image_path = os.path.join(input_path, filepath.file_upload.name)
        if not os.path.exists(image_path):
            return HttpResponse(f"File not found at path: {image_path}", status=404)
        image = cv2.imread(image_path)
        #print(image)
        if image is None:
            return HttpResponse('Could not read image', status=400)
        _, img_encoded = cv2.imencode('.jpg', image)
        # img_bytes = img_encoded.tobytes()
        
        inpt_arr=[]
        inpt1_arr=[]
        def rdraw_lines_with_text(image, landmarks, landmark_pairs):
            reference_real_world_size = 3.5

            for pair in landmark_pairs:
                start_idx = pair['start']
                end_idx = pair['end']
                start_pt = landmarks[start_idx]
                end_pt = landmarks[end_idx]
                distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
                inpt_arr.append(float(f"{distance:.3f}"))

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
                start_pt = landmarks[start_idx]
                end_pt = landmarks[end_idx]
                distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
                inpt1_arr.append(float(f"{distance:.3f}"))

                #Draw the line
                cv2.line(image, start_pt, end_pt, (0, 255, 0), 2)

                # Calculate the midpoint
                midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

                # Adjust the position of the text to be above the line
                text_position = (midpoint[0], midpoint[1] - 10)

                #Annotate with the label
            # cv2.putText(image, str(w), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return image


        def apply_filter(image, landmarks, landmark_indices):

            for idx in landmark_indices:
                landmark_pt = tuple(map(int, landmarks[idx]))
                cv2.circle(image, landmark_pt, 3, (0, 255, 0), -1)
            return image
        
        # inpt_arr.clear()
        # inpt1_arr.clear()
        ratios1=[[],[],[]]
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        # return image,rft_arr,rft1_arr
        print(inpt_arr)
        print(inpt1_arr)
        
        for i, j,r1,r2 in zip(inpt_arr,inpt1_arr,Name,Name2):
            p=i*1.618
            q=p/j
         
            if q*100>100:
                q=j/p
                q*100
                ratios1[0].append({
                'Name': f"{r1}, {r2}",
                'dist1': f"{i:.3f}",
                'dist2': f"{j:.3f}",
                'Percentage': f"{q * 100:.3f}"
                })
                a.append(q*100)
                # print(f"{r1},{r2}\n{i,j},percentage= {q*100}\n")
               
            else:
                q*100
                ratios1[0].append({
                'Name': f"{r1}, {r2}",
                'dist1': f"{i:.3f}",
                'dist2': f"{j:.3f}",
                'Percentage': f"{q * 100:.3f}"
                })
                a.append(q*100)
            avg = sum(a)/ len(a)
            avg= f"{avg:.3f}"
        ratios1[1].append({'average1':avg})
                # print(f"{r1},{r2}\n{i,j},percentage= {q*100}\n")
                
        #print(ratios1)
        avg = sum(a)/ len(a)
        print(f"{avg:.3f}")
        


        output_path = os.path.join(folder_path1,ids+'.jpg')
        cv2.imwrite(output_path, image)
        filepath = UploadedImage.objects.get(id=ids)
        filepath.processed_image= output_path
        ratios1[2].append({'output_image': output_path})
        print(ratios1)


        inpt_ref_json='INPT_REF_json'
        json_path1=os.path.join(inpt_ref_json,f'inpt_ref{ids}.json')
        print("JOSN",json_path1)
        filepath.processed_json= json_path1

        print(output_path)
        filepath.save()
        # cv2.imshow("input_as_reference", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        with open (json_path1,'w',encoding="utf-16") as f:
            
                # return f.write(res_json) 
            print("fcre",ratios1) 
            json.dump(ratios1,f)
        response = Response(ratios1, content_type='application/json')
        response.accepted_renderer = JSONRenderer()
        response.accepted_media_type = 'application/json'
        response.renderer_context = {}

        return HttpResponse("response")
     
class INPTasREF_retrive(View):
    def get(self, request, ids, *args, **kwargs):
        try:
            # Retrieve the file path from the database using the provided id
            filepath = UploadedImage.objects.get(id=ids)
            processed_json = filepath.processed_json
            
            # Construct the full file path
            file_path = os.path.join(processed_json)
            
            # Open and load the JSON file
            with open(file_path, 'r', encoding="utf-16") as jsonfile:
                json_data = json.load(jsonfile)
            
            # Return the JSON data in the response
            response = Response(json_data, content_type='application/json')
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = 'application/json'
            response.renderer_context = {}
            
            return response
        
        except UploadedImage.DoesNotExist:
            return JsonResponse({"error": "Image not found"}, status=404)
        except FileNotFoundError:
            return JsonResponse({"error": "File not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

        
class INPTasREF_pdf(View):
    def get(self, request, ids, *args, **kwargs):
            
        # Retrieve the file path from the database using the provided id
        filepath = UploadedImage.objects.get(id=ids)
        processed_json = filepath.processed_json
        
        # Construct the full file path
        file_path = os.path.join(processed_json)
        
        # Open and load the JSON file
        with open(file_path, 'r', encoding="utf-16") as jsonfile:
            json_data = json.load(jsonfile)
            print("J1",json_data)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Define styles
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Input as Reference Report", styles['Title']))
        elements.append(Spacer(1, 8))

        image_paths = []
        avg_paths = []
        # custom_style = styles.add(ParagraphStyle(name='CustomStyle', fontSize=12, textColor=colors.red))
# Check if json_data is a list or tuple and has more than 2 items
        if isinstance(json_data, (tuple, list)) and len(json_data) >= 3:
            try:
                    # Extract image paths from the nested dictionaries
                    image_paths = [
                        json_data[2][0].get('output_image')
                    ]
                    print("Image paths:", image_paths)  # Printing extracted image paths
            except (KeyError, IndexError) as e:
                    print("Error accessing image paths:", e)

    # Process each image path
        for i,img_path in enumerate(image_paths):
            if img_path:
                elements.append(Image(img_path, width=440, height=563))

                if i == 0: 
                     elements.append(Spacer(1, 12)) 
                     elements.append(Paragraph("Ploted Coodinates of Input as Reference", styles['Title']))
                     
               

        if isinstance(json_data, (tuple, list)) and len(json_data) >= 3:
            try:
                    # Extract image paths from the nested dictionaries
                    avg_paths = [
                        json_data[1][0].get('average1')
                    ]
                    print("Image paths:", avg_paths)  # Printing extracted image paths
            except (KeyError, IndexError) as e:
                    print("Error accessing image paths:", e)
                      
            for i,avg_path in enumerate(avg_paths):
             if avg_path:
                  elements.append(Paragraph(f"Average Percentage: {avg_path}%", styles['Title']))
                  elements.append(Spacer(1, 24))

        # Handle table data safely
        data_for_table = [['S.No.', 'Name', 'Distance', 'Percentage']]
        # if len(json_data) > 3 and isinstance(json_data[0], list):
        for i, ratio in enumerate(json_data[0]):
                if isinstance(ratio, dict):
                    description1 = ratio.get('Name', 'N/A')
                    description2 = ratio.get('Description2', 'N/A')
                    distance1 = ratio.get('dist1', 'N/A')
                    distance2 = ratio.get('dist2', 'N/A')
                    percentage = ratio.get('Percentage', 'N/A')

                    if description2 != 'N/A':
                        description = f"{description1} - {description2}"
                    else:
                        description = description1

                    data_for_table.append([
                        i + 1,
                        description,
                        f"{distance1} - {distance2}",
                        f"{percentage}%"
                    ])
                    
                    
            # print("DGSGSSUK")
        # Define column widths
        col_widths = [doc.width / 14, doc.width / 1.3, doc.width / 5, doc.width / 9]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)
        d=doc.build(elements)
        buffer.seek(0)
        print(d)

        filepath = UploadedImage.objects.get(id=ids)
        inpt_json = 'pdf_inpt_as_ref/'
        pdf_path = os.path.join(inpt_json, f'inpt{ids}.pdf')
        filepath.processed_pdf= pdf_path
        filepath.save() 

        with open(pdf_path, 'wb') as f:
            f.write(buffer.getvalue())

        # Return PDF as HTTP response
        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="phi{ids}.pdf"'
        return response
    
        # return HttpResponse(response)



class phi_matrix(View):
    def get(self, request, ids, *args, **kwargs):
        input_path = r'/input_image/'
        try:
            filepath = UploadedImage.objects.get(id=ids)
            print("image->",filepath)
        except UploadedImage.DoesNotExist:
            return HttpResponse('Image not found', status=404)
        if 'input_image' in filepath.file_upload.name:
            image_path = filepath.file_upload.path 
            print("file->",image_path)
        else:
            image_path = os.path.join(input_path, filepath.file_upload.name)
        if not os.path.exists(image_path):
            return HttpResponse(f"File not found at path: {image_path}", status=404)
        image = cv2.imread(image_path)
        #print(image)
        if image is None:
            return HttpResponse('Could not read image', status=400)
        _, img_encoded = cv2.imencode('.jpg', image)
        # img_bytes = img_encoded.tobytes()    
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        global_perc = []
        details = []
        phi_ratios = [[],[],[]]


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
        def parse_details(details):

            for detail in details:
                try:
                    if detail.startswith('#'):
                        # Format with headers (e.g., '#Face_Ratio')
                        header = detail.split('\n')[0].strip('#').strip()
                        distances = re.search(r'Distance:\(([^)]+)\)', detail).group(1).split(',')
                        percentage = re.search(r'Percentage: ([\d.]+)', detail).group(1)

                        # Clean and convert distances
                        dist1 = float(distances[0].strip().strip('()'))
                        dist2 = float(distances[1].strip().strip('()'))
                        ratio_per = float(percentage)

                        phi_ratios[0].append({
                            'Description1': header,
                            #'Description2': 'N/A',
                            'dist1': dist1,
                            'dist2': dist2,
                            'Percentage': ratio_per
                        })
                    else:
                        # Format without headers (e.g., 'Nosetip to centre of lips')
                        lines = detail.split('\n')
                        desc1 = lines[0].strip()
                        desc2 = lines[1].strip()
                        distances = re.search(r'Distance:\(([^)]+)\)', detail).group(1).split(',')
                        percentage = re.search(r'Percentage: ([\d.]+)', detail).group(1)

                        # Clean and convert distances
                        dist1 = float(distances[0].strip().strip('()'))
                        dist2 = float(distances[1].strip().strip('()'))
                        ratio_per = float(percentage)

                        phi_ratios[0].append({
                            'Description1': desc1,
                            'Description2': desc2,
                            'dist1': dist1,
                            'dist2': dist2,
                            'Percentage': ratio_per
                        })
                       
         
                except (AttributeError, ValueError) as e:
                    # Handle cases where extraction or conversion fails
                    print(f"Parsing failed for detail: {detail}. Error: {e}")
            #     avg = sum(global_perc) / len(global_perc)*100 if global_perc else 0
            #     avg = avg*0.01
            # phi_ratios[1].append({'average2':f'{avg}'}) 

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
            parse_details(details)

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

        fun(image, landlist)
        print(len((details)))
        print(len(phi_ratios))
        
     


        
       
        avg = sum(global_perc) / len(global_perc)*100 if global_perc else 0
        avg = avg*0.01
        avg=f"{avg:.3f}"
        phi_ratios[1].append({'average2':f'{avg}'})
        print(avg)
        
        output_path = os.path.join(folder_path2,ids+'.jpg')
        cv2.imwrite(output_path, image)
        filepath = UploadedImage.objects.get(id=ids)
        filepath.phi_matrix_image= output_path
        phi_ratios[2].append({
                        'output_image': output_path,
                    })
        print(phi_ratios)
                       
        phi_json='PHI_json/'
        json_path2=os.path.join(phi_json,f'phi{ids}.json')
        filepath.phi_matrix_json= json_path2
        print(output_path)
        filepath.save()
        with open (json_path2,'w',encoding="utf-16") as f:
            
                # return f.write(res_json) 
            print("fcre",phi_ratios) 
            json.dump(phi_ratios,f)
        response = Response(phi_ratios, content_type='application/json')
        response.accepted_renderer = JSONRenderer()
        response.accepted_media_type = 'application/json'
        response.renderer_context = {}

        return HttpResponse("response")
     
class PHI_retrive(View):
    def get(self, request, ids, *args, **kwargs):
        try:
            # Retrieve the file path from the database using the provided id
            filepath = UploadedImage.objects.get(id=ids)
            phi_matrix_json = filepath.phi_matrix_json
            
            # Construct the full file path
            file_path = os.path.join(phi_matrix_json)
            
            # Open and load the JSON file
            with open(file_path, 'r', encoding="utf-16") as jsonfile:
                json_data = json.load(jsonfile)
            
            # Return the JSON data in the response
            response = Response(json_data, content_type='application/json')
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = 'application/json'
            response.renderer_context = {}
            
            return response
        
        except UploadedImage.DoesNotExist:
            return JsonResponse({"error": "Image not found"}, status=404)
        except FileNotFoundError:
            return JsonResponse({"error": "File not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
        
class PHI_pdf(View):
    def get(self, request, ids, *args, **kwargs):

        # Retrieve the file path from the database using the provided id
        filepath = UploadedImage.objects.get(id=ids)
        phi_matrix_json = filepath.phi_matrix_json
        
        # Construct the full file path
        file_path = os.path.join(phi_matrix_json)
        
        # Open and load the JSON file
        with open(file_path, 'r', encoding="utf-16") as jsonfile:
            json_data = json.load(jsonfile)
            print("J1",json_data)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Define styles
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Phi Matrix Report", styles['Title']))
        elements.append(Spacer(1, 8))

        image_paths = []
        avg_paths = []
        # custom_style = styles.add(ParagraphStyle(name='CustomStyle', fontSize=12, textColor=colors.red))
# Check if json_data is a list or tuple and has more than 2 items
        if isinstance(json_data, (tuple, list)) and len(json_data) >= 3:
            try:
                    # Extract image paths from the nested dictionaries
                    image_paths = [
                        json_data[2][0].get('output_image')
                    ]
                    print("Image paths:", image_paths)  # Printing extracted image paths
            except (KeyError, IndexError) as e:
                    print("Error accessing image paths:", e)

    # Process each image path
        for i,img_path in enumerate(image_paths):
            if img_path:
                elements.append(Image(img_path, width=440, height=563))

                if i == 0: 
                     elements.append(Spacer(1, 12)) 
                     elements.append(Paragraph("Ploted Coodinates of Phi Matrix", styles['Title']))
                     
               

        if isinstance(json_data, (tuple, list)) and len(json_data) >= 3:
            try:
                    # Extract image paths from the nested dictionaries
                    avg_paths = [
                        json_data[1][0].get('average2')
                    ]
                    print("Image paths:", avg_paths)  # Printing extracted image paths
            except (KeyError, IndexError) as e:
                    print("Error accessing image paths:", e)
                      
            for i,avg_path in enumerate(avg_paths):
             if avg_path:
                  elements.append(Paragraph(f"Average Percentage: {avg_path}%", styles['Title']))
                  elements.append(Spacer(1, 24))

        # Handle table data safely
        data_for_table = [['S.No.', 'Description', 'Distance', 'Percentage']]
        # if len(json_data) > 3 and isinstance(json_data[0], list):
        for i, ratio in enumerate(json_data[0]):
                if isinstance(ratio, dict):
                    desc1 = ratio.get('Description1', 'N/A')
                    desc2 = ratio.get('Description2', 'N/A')
                    description = f"{desc1} -\n {desc2}" if desc2 != 'N/A' else desc1
                    dist1 = ratio.get('dist1', 'N/A')
                    dist2 = ratio.get('dist2', 'N/A')
                    percentage = ratio.get('Percentage', 'N/A')
                    data_for_table.append([
                        i + 1,
                        description,
                        f"{dist1} - {dist2}",
                        f"{percentage}%"
                         
                    ])
                    
                    
            # print("DGSGSSUK")
        # Define column widths
        col_widths = [doc.width / 14, doc.width / 1.3, doc.width / 5, doc.width / 9]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)
        d=doc.build(elements)
        buffer.seek(0)
        print(d)

        filepath = UploadedImage.objects.get(id=ids)
        phi_json = 'pdf_phi/'
        pdf_path = os.path.join(phi_json, f'phi{ids}.pdf')
        filepath.phi_pdf= pdf_path
        filepath.save() 

        with open(pdf_path, 'wb') as f:
            f.write(buffer.getvalue())

        # Return PDF as HTTP response
        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="phi{ids}.pdf"'
        return response
    
        # return HttpResponse(response)

        
      




        # def json_data():
        #     with open (json_path,'r') as f:
        #         return f.read()  
        # s=json_data()
        # #res_json=json.loads(s)

        # print("type",type(s))


        # return HttpResponse(s)

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
class sym_assym(View):

    def get(self, request, ids, *args, **kwargs):
        input_path = r'/input_image/'
        try:
            filepath = UploadedImage.objects.get(id=ids)
            print("image->",filepath)
        except UploadedImage.DoesNotExist:
            return HttpResponse('Image not found', status=404)
        if 'input_image' in filepath.file_upload.name:
            image_path = filepath.file_upload.path 
            print("file->",image_path)
        else:
            image_path = os.path.join(input_path, filepath.file_upload.name)
        if not os.path.exists(image_path):
            return HttpResponse(f"File not found at path: {image_path}", status=404)
        image = cv2.imread(image_path)
        #print(image)
        if image is None:
            return HttpResponse('Could not read image', status=400)
        _, img_encoded = cv2.imencode('.jpg', image)
        # img_bytes = img_encoded.tobytes()       
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
            print(ans)

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
            print('a1',a1)
            print('a2',a2)
            print('a3',a3)
            per=a1+a2+a3
            return per

        def values(index, per,landlist,image):
            for idx, landmark_pt in enumerate(landlist):
                #cv2.circle(image, tuple(map(int, landmark_pt)), 1, (255, 0, 0), -1)

                for ind, an in zip(index,per):
                    if idx == ind:
                        cv2.putText(image, f'{an:.2f}', (landmark_pt[0], landmark_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return image
        
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

        sym_ratios = [[],[],[],[]]
        for result in combined_results:
            sym_ratios[0].append({
                "Name": result[0],
                "Distance": f"{result[1]:.3f}",
                "Symmetry": result[2] if len(result) > 2 else None,
                "Percentage": f"{result[3]:.3f}" if len(result) > 2 else None
            })
      
            
        
        #print(ratios)

        image_copy1 = image.copy()
        image_copy2 = image.copy()
        image_copy3 = image.copy()

        img11 = values(index1, ans, landlist, image_copy1)
        img11 = line(img11)

        img12 = values(index2, per, landlist, image_copy2)

        img13 = values(index2, per, landlist, image_copy3)
        img13 = line(img13)

        output_path = os.path.join(folder_path3,ids+'.jpg')
        cv2.imwrite(output_path, img11)

        output_path1 = os.path.join(folder_path4,ids+'.jpg')
        cv2.imwrite(output_path1, img12)

        output_path2 = os.path.join(folder_path5,ids+'.jpg')
        cv2.imwrite(output_path2, img13)
        
        sym_ratios[1].append({
            'output_image':output_path,
            # 'output_image1':output_path1,
            # 'output_image2':output_path2
            })
        sym_ratios[2].append({
        
         'output_image1':output_path1,
        })


            
        sym_ratios[3].append({
            'output_image2':output_path2
        })




        print("SYM",type(sym_ratios))

       

        filepath = UploadedImage.objects.get(id=ids)

        filepath.sym_proportion_image= output_path
        filepath.sym_unified_line_image= output_path1
        filepath.sym_unified_image= output_path2

        sym_json='SYM_ASYM_json/'
        json_path3=os.path.join(sym_json,f'sym{ids}.json')
        filepath.symmetric_json= json_path3

                   
        print(output_path)
        filepath.save()


        with open (json_path3,'w',encoding="utf-16") as f:
                
                # return f.write(res_json) 
            print("fcre",type(sym_ratios)) 
            json.dump(sym_ratios,f)

#modifications
        # with open(json_path3, 'r', encoding="utf-16") as jsonfile:
        #     data = json.load(jsonfile)
        #     print("JSIN DATA",data)

        # buffer = BytesIO()
        # doc = SimpleDocTemplate(buffer, pagesize=letter)
        # elements = []

        # # Define styles
        # styles = getSampleStyleSheet()

        # elements.append(Paragraph("Symmetric Asymmetric Report", styles['Title']))

        # # image_paths = [sym_ratios[1],sym_ratios[2],sym_ratios[3]]
        # # print("IMG",image_paths)

        # image_paths = []
        # if isinstance(sym_ratios, (tuple, list)) and len(sym_ratios) > 2:
        #     image_paths = [sym_ratios[1],sym_ratios[2],sym_ratios[3]]
        #     #print("image:",image_paths)  # Assume first three are image paths

        # for img_path in image_paths:
        #     if isinstance(img_path, str):
        #         elements.append(Image(img_path, width=400, height=500))
        #         elements.append(Spacer(1, 12))
        # # Handle table data safely
        # data_for_table = [['S.No.', 'Name', 'Distance', 'Symmetry', 'Percentage']]
        # if len(sym_ratios) > 3 and isinstance(sym_ratios[0], list):
        #     for i, ratio in enumerate(sym_ratios[0]):
        #         if isinstance(ratio, dict):
        #             data_for_table.append([
        #                 i + 1,
        #                 ratio.get('Name', 'N/A'),
        #                 (ratio.get('Distance', 0), 3),
        #                 ratio.get('Symmetry', 'N/A'),
        #                 f"{(ratio.get('Percentage', 0), 3)}%" if ratio.get('Percentage') else 'N/A'
        #             ])

        # # Define column widths
        # col_widths = [doc.width / 14, doc.width / 1.3, doc.width / 5, doc.width / 9]
        # table = Table(data_for_table, colWidths=col_widths)
        # table.setStyle(TableStyle([
        #     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        #     ('GRID', (0, 0), (-1, -1), 1, colors.black),
        #     ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
        #     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        #     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        #     ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        #     ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        #     ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        #     ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        #     ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        # ]))
        # elements.append(table)
        # d=doc.build(elements)
        # buffer.seek(0)
        # print(d)
        
        # sym_json = 'pdf_sym/'
        # pdf_path = os.path.join(sym_json, f'sym{ids}.pdf')

        # with open(pdf_path, 'wb') as f:
        #     f.write(buffer.getvalue())

        # # Return PDF as HTTP response
        # response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        # response['Content-Disposition'] = f'attachment; filename="sym{ids}.pdf"'
        # #return response
    
                    
                    


        response = Response(sym_ratios, content_type='application/json')
        response.accepted_renderer = JSONRenderer()
        response.accepted_media_type = 'application/json'
        response.renderer_context = {}
        

        return HttpResponse("response")
        #return response
       
         
    
class SYM_retrive(View):
    def get(self, request, ids, *args, **kwargs):
        
        try:
            # Retrieve the file path from the database using the provided id
            filepath = UploadedImage.objects.get(id=ids)
            symmetric_json = filepath.symmetric_json
            
            # Construct the full file path
            file_path = os.path.join(symmetric_json)
            
            # Open and load the JSON file
            with open(file_path, 'r', encoding="utf-16") as jsonfile:
                json_data = json.load(jsonfile)
            print("J1",type(json_data))
            
            # Return the JSON data in the response
            response = Response(json_data, content_type='application/json')
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = 'application/json'
            response.renderer_context = {}

            # image_paths = [sym_ratios[1],sym_ratios[2],sym_ratios[3]]
            # print("IMG",image_paths)

            return response
        
        except UploadedImage.DoesNotExist:
            return JsonResponse({"error": "Image not found"}, status=404)
        except FileNotFoundError:
            return JsonResponse({"error": "File not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    

class SYM_report(View):

    def get(self, request, ids, *args, **kwargs):
# Retrieve the file path from the database using the provided id
        filepath = UploadedImage.objects.get(id=ids)
        symmetric_json = filepath.symmetric_json
        
        # Construct the full file path
        file_path = os.path.join(symmetric_json)
        
        # Open and load the JSON file
        with open(file_path, 'r', encoding="utf-16") as jsonfile:
            json_data = json.load(jsonfile)
        print("J1",json_data)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Define styles
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Symmetric Asymmetric Report", styles['Title']))
        elements.append(Spacer(1, 8))

        image_paths = []
        # custom_style = styles.add(ParagraphStyle(name='CustomStyle', fontSize=12, textColor=colors.red))
# Check if json_data is a list or tuple and has more than 2 items
        if isinstance(json_data, (tuple, list)) and len(json_data) >= 3:
            try:
                    # Extract image paths from the nested dictionaries
                    image_paths = [
                        json_data[1][0].get('output_image'),
                        json_data[2][0].get('output_image1'),
                        json_data[3][0].get('output_image2')
                    ]
                    print("Image paths:", image_paths)  # Printing extracted image paths
            except (KeyError, IndexError) as e:
                    print("Error accessing image paths:", e)

    # Process each image path
        for i,img_path in enumerate(image_paths):
            if img_path:
                elements.append(Image(img_path, width=440, height=563))
                
                if i == 0: 
                     elements.append(Spacer(1, 12)) 
                     elements.append(Paragraph("Symmetric Proportion", styles['Title']))
                     
                elif i == 1:
                     elements.append(Spacer(1, 12))  
                     elements.append(Paragraph("Symmetric Unified", styles['Title']))
                     
                elif i == 2:
                     elements.append(Spacer(1, 12))  
                     elements.append(Paragraph("Symmetric Unified Lines ", styles['Title']))
                elements.append(Spacer(1, 24))


        # Handle table data safely
        data_for_table = [['S.No.', 'Name', 'Distance', 'Symmetry', 'Percentage']]
        # if len(json_data) > 3 and isinstance(json_data[0], list):
        for i, ratio in enumerate(json_data[0]):
                if isinstance(ratio, dict):
                    data_for_table.append([
                        i + 1,
                        ratio.get('Name', 'N/A'),
                        ratio.get('Distance', 0),
                        ratio.get('Symmetry')if ratio.get('Symmetry') else 'N/A',
                        f"{ratio.get('Percentage', 0)}%" if ratio.get('Percentage') else 'N/A',
                         
                    ])
                    
                    
            # print("DGSGSSUK")
        # Define column widths
        col_widths = [doc.width / 14, doc.width / 1.3, doc.width / 5, doc.width / 9]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)
        d=doc.build(elements)
        buffer.seek(0)
        print(d)

        filepath = UploadedImage.objects.get(id=ids)
        sym_pdf = 'pdf_sym/'
        pdf_path = os.path.join(sym_pdf, f'sym{ids}.pdf')
        filepath.symmetric_pdf= pdf_path
        filepath.save()
        

        with open(pdf_path, 'wb') as f:
            f.write(buffer.getvalue())

        # Return PDF as HTTP response
        response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="sym{ids}.pdf"'
        return response
    
        # return HttpResponse(response)
    








   
    

    
            












