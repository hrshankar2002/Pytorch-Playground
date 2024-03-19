# Generated by Django 4.2.11 on 2024-03-19 04:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Classifier',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('samples', models.IntegerField()),
                ('classes', models.IntegerField()),
                ('features', models.IntegerField()),
                ('seed', models.IntegerField()),
                ('dataset_type', models.CharField(max_length=20)),
                ('epochs', models.IntegerField()),
                ('lr', models.FloatField()),
                ('hidden_features', models.IntegerField()),
                ('in_features', models.IntegerField()),
                ('out_features', models.IntegerField()),
                ('activation', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='Regressor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('epochs', models.IntegerField()),
                ('lr', models.FloatField()),
                ('samples', models.IntegerField()),
                ('test_size', models.FloatField()),
                ('random_state', models.IntegerField()),
                ('noise', models.IntegerField()),
            ],
        ),
    ]
