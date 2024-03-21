from django.contrib import admin
from .models import Classifier, Regressor

admin.site.register(Regressor)
admin.site.register(Classifier)