from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import fileUploadView, golden_ratio, input_as_reference, phi_matrix,sym_assym, GR_retrive, INPT_REF_retrive

router = DefaultRouter()
router.register(r'Uploadfile', fileUploadView)


urlpatterns = [  
    path('', include(router.urls)),
    path("golden_ratio/<ids>/", golden_ratio.as_view(),name="golden_ratio"),
    path("input_as_reference/<ids>/", input_as_reference.as_view(),name="input_as_reference"),
    path("phi_matrix/<ids>/", phi_matrix.as_view(),name="phi_matrix"),
    path("sym/<ids>/", sym_assym.as_view(),name="sym"),
    path("gr_json/<ids>/", GR_retrive.as_view(),name="gr_json"),
    path("inpt_ref_json/<ids>/", INPT_REF_retrive.as_view(),name="inpt_ref_json"),
    # path("phi_json/<ids>/", GR_retrive.as_view(),name="phi_json"),
    # path("symt_json/<ids>/", GR_retrive.as_view(),name="symt_json"),


]














