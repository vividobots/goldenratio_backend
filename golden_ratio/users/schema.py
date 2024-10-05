import graphene
from graphene_django import DjangoObjectType
from graphene.relay.node import Node
from django.db.models import Q

from .models import (
    User
 )

class UserType(DjangoObjectType):
    class Meta:
        model = User
        
class UsersQuery(object):
    all_users_detail = graphene.List(UserType)
    get_user_detail = graphene.Field(UserType, id=graphene.String())
    
    def resolve_all_users_detail(self, info, **kwargs):
        return User.objects.all()
    
    def resolve_get_user_detail(self, info, id,**kwargs):
        _, raw_pk = Node.from_global_id(id)
        return User.objects.get(id=raw_pk)
    
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
        
            
class UsersMutation(graphene.ObjectType):
    update_user_detail = UpdateProfileDetails.Field()
