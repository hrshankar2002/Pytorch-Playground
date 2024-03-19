from rest_framework.serializers import ModelSerializer

from .models import Classifier, Regressor


class ClassifierSerializer(ModelSerializer):
    class Meta:
        model = Classifier
        fields = ('__all__')
        
class RegressorSerializer(ModelSerializer):
    class Meta:
        model = Regressor
        fields = ('__all__')