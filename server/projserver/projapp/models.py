from django.db import models


class Regressor(models.Model):
    epochs = models.IntegerField()
    lr = models.FloatField()
    samples = models.IntegerField()
    test_size = models.FloatField()
    random_state = models.IntegerField()
    noise = models.IntegerField()
    
class Classifier(models.Model):
    samples = models.IntegerField()
    classes = models.IntegerField()
    features = models.IntegerField()
    seed = models.IntegerField()
    dataset_type = models.CharField(max_length=20)
    epochs = models.IntegerField()
    lr = models.FloatField()
    hidden_features = models.IntegerField()
    in_features = models.IntegerField()
    out_features = models.IntegerField()
    activation = models.CharField(max_length=20)