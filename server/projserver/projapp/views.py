from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .helper import ClassifierPipeline, RegressionPipeline
from .models import Classifier, Regressor
from .serializers import ClassifierSerializer, RegressorSerializer


@api_view(['GET'])
def get_params_reg(request):
    params = Regressor.objects.all()
    serializer = RegressorSerializer(params, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def get_params_class(request):
    params = Classifier.objects.all()
    serializer = ClassifierSerializer(params, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def clear_post_params_reg(request):
    Regressor.objects.all().delete()
    serializer = RegressorSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return JsonResponse({'message': 'Data processed successfully'}, status=200)
    else:
        return JsonResponse({'error': 'Incorrect format'}, status=400)
    
@api_view(['POST'])
def clear_post_params_class(request):
    Classifier.objects.all().delete()
    serializer = ClassifierSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return JsonResponse({'message': 'Data processed successfully'}, status=200)
    else:
        return JsonResponse({'error': 'Incorrect format'}, status=400)
    


   