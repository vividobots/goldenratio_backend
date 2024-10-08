from rest_framework import serializers
from users.models import UploadedImage

class fileUploadSerializers(serializers.ModelSerializer):
    class Meta: 
        model = UploadedImage
        fields = ['id','file_upload','user']

    