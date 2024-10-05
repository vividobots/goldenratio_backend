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

        