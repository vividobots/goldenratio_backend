import graphene
from graphql_auth.schema import UserQuery, MeQuery
from graphql_auth import mutations
from users.schema import UsersQuery,UsersMutation, FileUploadQuery

                              
# from users.schema import UserstMutations

class AuthMutation(graphene.ObjectType):
   register = mutations.Register.Field()
   verify_account = mutations.VerifyAccount.Field()
   token_auth = mutations.ObtainJSONWebToken.Field()
   update_account = mutations.UpdateAccount.Field()
   resend_activation_email = mutations.ResendActivationEmail.Field()
   send_password_reset_email = mutations.SendPasswordResetEmail.Field()
   password_reset = mutations.PasswordReset.Field()

class Query(UserQuery,FileUploadQuery,
            MeQuery,UsersQuery,
            graphene.ObjectType):
    pass

class Mutation(AuthMutation,UsersMutation,
               graphene.ObjectType):
   pass

schema = graphene.Schema(query=Query, mutation=Mutation)