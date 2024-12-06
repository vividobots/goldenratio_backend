import graphene
from graphene_django import DjangoObjectType
from graphene.relay.node import Node
from django.db.models import Q
from .models import UploadedImage

from .models import (
    User,UploadedImage
 )

class UserType(DjangoObjectType):
    class Meta:
        model = User
class FileUploadType(DjangoObjectType):
    class Meta:
        model = UploadedImage

        
class UsersQuery(object):
    all_users_detail = graphene.List(UserType)
    get_user_detail = graphene.Field(UserType, id=graphene.String())
    
    def resolve_all_users_detail(self, info, **kwargs):
        return User.objects.all()
    
    def resolve_get_user_detail(self, info, id,**kwargs):
        _, raw_pk = Node.from_global_id(id)
        return User.objects.get(id=raw_pk)
    
class FileUploadQuery(object):
    all_files_detail = graphene.List(FileUploadType)
    get_files_by_user_detail = graphene.List(FileUploadType, id=graphene.String())
    image_get_by_id= graphene.Field(FileUploadType, id=graphene.String())


    def resolve_all_files_detail(self, info, **kwargs):
        return UploadedImage.objects.all()

    def resolve_get_files_by_user_detail(self, info, id,**kwargs):
        _, raw_pk = Node.from_global_id(id)
        return UploadedImage.objects.filter(user_id=raw_pk)   
    
    def resolve_image_get_by_id(self, info, id,**kwargs):
        return UploadedImage.objects.get(id=id)  

    
    
class UpdateProfileDetailsInput(graphene.InputObjectType):
    id = graphene.String()
    name = graphene.String()
    number = graphene.String()
    
class UpdateProfileDetails(graphene.Mutation):
    users=graphene.Field(UserType)

    class Arguments: 
        input = UpdateProfileDetailsInput(required=True)
        
        
    def mutate(self, info, input):  
        _, raw_pk = Node.from_global_id(input.id)
  
        data = User.objects.get(id=raw_pk)
        data.username = input.name
        data.contactno = input.number
        data.save()
        return UpdateProfileDetails(users=data)    
        

class DeleteImagebyId(graphene.Mutation):
    class Arguments:
        id = graphene.String(required=True) 
    success = graphene.Boolean()
    def mutate(self, info, id):
        try:
            image = UploadedImage.objects.get(id=id)
            image.delete()
            return DeleteImagebyId(success=True)
        except UploadedImage.DoesNotExist:
            return DeleteImagebyId(success=False)
                    
class UsersMutation(graphene.ObjectType):
    update_user_detail = UpdateProfileDetails.Field()
    delete_uploaded_image = DeleteImagebyId.Field()
    
    

