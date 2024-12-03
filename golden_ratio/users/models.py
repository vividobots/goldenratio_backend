from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.postgres.fields import ArrayField

class User(AbstractUser):
    email = models.EmailField(blank=False, max_length=50, verbose_name="email address")
    contactno = models.CharField(max_length=25, blank=True)
    lastUpdate = models.DateTimeField(auto_now=True, blank=True)

    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name=('groups'),
        blank=True,
        related_name='customuser_set', 
        related_query_name='customuser',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name=('user permissions'),
        blank=True,
        related_name='customuser_set', 
        related_query_name='customuser',  
    )

    USERNAME_FIELD = "username"


class UploadedImage(models.Model):
    user=models.ForeignKey("User", on_delete=models.CASCADE)
    file_upload = models.ImageField(upload_to="input_image", blank=True)
    reference=models.CharField(blank=True, max_length=100)
    processed_image=models.CharField(blank=True, max_length=100)
    phi_matrix_image=models.CharField(blank=True, max_length=100)
    sym_proportion_image=models.CharField(blank=True, max_length=100)
    sym_unified_image=models.CharField(blank=True, max_length=100)
    sym_unified_line_image=models.CharField(blank=True, max_length=100)
    reference_json=models.CharField(blank=True, max_length=100)
    reference_pdf=models.CharField(blank=True, max_length=100)
    processed_json=models.CharField(blank=True, max_length=100)
    processed_pdf=models.CharField(blank=True, max_length=100)
    phi_matrix_json=models.CharField(blank=True, max_length=100)
    phi_pdf=models.CharField(blank=True, max_length=100)
    symmetric_json=models.CharField(blank=True, max_length=100)
    symmetric_pdf=models.CharField(blank=True, max_length=100)
    shrink_image=models.CharField(blank=True, max_length=100)
    bulge_image=models.CharField(blank=True, max_length=100)
    status=models.BooleanField(default=False)
    lastUpdate = models.DateTimeField(auto_now=True, blank=True)
