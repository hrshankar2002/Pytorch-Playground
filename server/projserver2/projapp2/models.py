from django.db import models


class Regressor(models.Model):
    epochs = models.IntegerField()
    lr = models.FloatField()
    samples = models.IntegerField()
    test_size = models.FloatField()
    random_state = models.IntegerField()
    noise = models.IntegerField()
    
    def __str__(self):
        return f"RegressorPipeline: {self.id}"
    
class Classifier(models.Model):
    epochs = models.IntegerField()
    dataset_type = models.CharField(max_length=100)
    test_size = models.FloatField()
    lr = models.FloatField()
    in_features = models.IntegerField()
    hidden_features = models.IntegerField()
    out_features = models.IntegerField()
    samples = models.IntegerField()
    seed = models.IntegerField()
    noise = models.FloatField()
    classes = models.IntegerField()
    activation = models.CharField(max_length=100)
    layer_count = models.IntegerField()

    def __str__(self):
        return f"ClassifierPipeline: {self.id}"