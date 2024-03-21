import os

import pyrebase
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, JsonResponse
from dotenv import load_dotenv
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .firebase.firebase_helper import uploadimage
from .helper import ClassifierPipeline, RegressionPipeline
from .models import Classifier, Regressor
from .serializers import ClassifierSerializer, RegressorSerializer

fs = FileSystemStorage()

load_dotenv()

API_KEY = os.getenv("API_KEY")
AUTH_DOMAIN = os.getenv("AUTH_DOMAIN")
PROJ_ID = os.getenv("PROJ_ID")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET")
SENDER_ID = os.getenv("SENDER_ID")
APP_ID = os.getenv("APP_ID")
SERV_ACCNT = os.getenv("SERV_ACCNT")
DB_URL = os.getenv("DB_URL")
DSIMG = os.getenv("DSIMG")
ENDIMG = os.getenv("ENDIMG")

config = {
    "apiKey": API_KEY,
    "authDomain": AUTH_DOMAIN,
    "projectId": PROJ_ID,
    "storageBucket": STORAGE_BUCKET,
    "messagingSenderId": SENDER_ID,
    "appId": APP_ID,
    "serviceAccount": SERV_ACCNT,
    "databaseURL": DB_URL
}

firebase_helper = pyrebase.initialize_app(config)
storage = firebase_helper.storage()

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
    
    if fs.exists(DSIMG):
        fs.delete(DSIMG)
    if fs.exists(ENDIMG):
        fs.delete(ENDIMG)
      
    serializer = RegressorSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        
        params = serializer.data
        
        train_loss, test_loss = RegressionPipeline(
            params["epochs"],
            params["lr"],
            params['samples'],
            params["test_size"],
            params["random_state"],
            params["noise"]
            ).run(DSIMG,ENDIMG)
        
        url1, url2 = uploadimage(storage, DSIMG, ENDIMG)
        
        return JsonResponse({'Dataset_Url': url1, 
                             "Reg_Url": url2,
                             'Train_Loss': train_loss,
                             'Test_Loss': test_loss
                             }, status=200)
    else:
        return JsonResponse({'error': 'Incorrect format'}, status=400)
    
@api_view(['POST'])
def clear_post_params_class(request):
    Classifier.objects.all().delete()
    
    if fs.exists(DSIMG):
        fs.delete(DSIMG)
    if fs.exists(ENDIMG):
        fs.delete(ENDIMG)
      
    serializer = ClassifierSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        
        params = serializer.data
        
        train_loss, test_loss = ClassifierPipeline(
            samples=(params['samples']),
            classes=(params['classes']),
            seed=(params['seed']),
            dataset_type=params['dataset_type'],
            epochs=params['epochs'],
            lr=(params['lr']),
            hidden_features=params['hidden_features'],
            in_features=params['in_features'],
            out_features=params['out_features'],
            activation=params['activation'],
            layer_count=params['layer_count'],
            test_size=float(params['test_size'])
            ).run(DSIMG,ENDIMG)
        
        url1, url2 = uploadimage(storage, DSIMG, ENDIMG)
        
        return JsonResponse({'Dataset_Url': url1, 
                             "Reg_Url": url2,
                             'Train_Loss': train_loss,
                             'Test_Loss': test_loss
                             }, status=200)
    else:
        return JsonResponse({'error': 'Incorrect format'}, status=400)
    