from django.urls import include, path
from rest_framework.routers import DefaultRouter
from .views import fileUploadView, golden_ratio, input_as_reference, phi_matrix,sym_assym, GR_retrive,GR_pdf, INPTasREF_retrive,INPTasREF_pdf, PHI_retrive, PHI_pdf, SYM_retrive, SYM_report,Shrin_Bulge

router = DefaultRouter()
router.register(r'Uploadfile', fileUploadView)


urlpatterns = [  
    path('', include(router.urls)),
    path("golden_ratio/<ids>/", golden_ratio.as_view(),name="golden_ratio"),
    path("input_as_reference/<ids>/", input_as_reference.as_view(),name="input_as_reference"),
    path("phi_matrix/<ids>/", phi_matrix.as_view(),name="phi_matrix"),
    path("sym/<ids>/", sym_assym.as_view(),name="sym"),
    path("gr_json/<ids>/", GR_retrive.as_view(),name="gr_json"),
    path("gr_pdf/<ids>/", GR_pdf.as_view(),name="gr_pdf"),
    path("inpt_json/<ids>/", INPTasREF_retrive.as_view(),name="inpt_json"),
    path("inpt_pdf/<ids>/", INPTasREF_pdf.as_view(),name="inpt_pdf"),
    path("phi_json/<ids>/", PHI_retrive.as_view(),name="phi_json"),
    path("phi_pdf/<ids>/", PHI_pdf.as_view(),name="phi_pdf"),
    path("sym_json/<ids>/", SYM_retrive.as_view(),name="sym_json"),
    path("sym_pdf/<ids>/", SYM_report.as_view(),name="sym_pdf"),
     path("strink_bulge/<ids>/", Shrin_Bulge.as_view(),name="strink_bulge"),


]














