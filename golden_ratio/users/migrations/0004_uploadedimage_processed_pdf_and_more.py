# Generated by Django 5.0.6 on 2024-09-28 06:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_uploadedimage_phi_pdf'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedimage',
            name='processed_pdf',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AddField(
            model_name='uploadedimage',
            name='reference_pdf',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]