import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_blobs, make_circles, make_regression
from sklearn.model_selection import train_test_split
from torch import nn


class ClassifierModelRelu(nn.Module):
    def __init__(self, 
                 input_features, 
                 output_features, 
                 hidden_units,
                 layer_count):
        super().__init__()
        
        self.layer_count = layer_count
        self.hidden_units = hidden_units
        
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU()
            )
        self.add_stack()
        self.linear_layer_stack.add_module('output layer', nn.Linear(hidden_units, output_features))
        
    def forward(self, X):
        return self.linear_layer_stack(X)
    
    def add_stack(self):
        for i in range(self.layer_count):
            self.linear_layer_stack.add_module(f'hidden_layer{i+1}', nn.Linear(self.hidden_units, self.hidden_units))
            self.linear_layer_stack.add_module(f'relu{i+1}', nn.ReLU())


class ClassifierModelSigmoid(nn.Module):
    def __init__(self, 
                 input_features, 
                 output_features, 
                 hidden_units,
                 layer_count):
        super().__init__()
        
        self.layer_count = layer_count
        self.hidden_units = hidden_units
        
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Sigmoid()
            )
        self.add_stack()
        self.linear_layer_stack.add_module('output layer', nn.Linear(hidden_units, output_features))
        
    def forward(self, X):
        return self.linear_layer_stack(X)
    
    def add_stack(self):
        for i in range(self.layer_count):
            self.linear_layer_stack.add_module(f'hidden_layer{i+1}', nn.Linear(self.hidden_units, self.hidden_units))
            self.linear_layer_stack.add_module(f'sigmoid{i+1}', nn.Sigmoid())


class ClassifierPipeline:
    def __init__(
        self,
        epochs=None,
        dataset_type=None,
        test_size=None,
        lr=None,
        in_features=None,
        hidden_features=None,
        out_features=None,
        samples=None,
        seed=None,
        noise=None,
        classes=None,
        activation=None,
        layer_count=None
    ):

        self.epochs = epochs
        self.dataset_type = dataset_type
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.classes = classes
        self.samples = samples
        self.seed = seed
        self.noise = noise
        self.lr = lr
        self.activation = activation
        self.test_size = test_size
        self.layer_count = layer_count
        self.model = None
        self.X_tensor = None
        self.y_tensor = None
        self.X = None
        self.y = None
        self.train_loss = 0
        self.test_loss = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def plot_decision_boundary(self,endimg):
        model = self.model

        X = self.X_test
        y = self.y_test

        model.to("cpu")
        X, y = X.to("cpu"), y.to("cpu")

        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101)
        )

        X_to_pred_on = torch.from_numpy(
            np.column_stack((xx.ravel(), yy.ravel()))
        ).float()

        model.eval()
        with torch.inference_mode():
            y_logits = model(X_to_pred_on)

        if len(torch.unique(y)) > 2:
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        else:
            y_pred = torch.round(torch.sigmoid(y_logits))

        y_pred = y_pred.reshape(xx.shape).detach().numpy()
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.savefig(endimg)
        plt.close()

    def make_blob_dataset_fig(self,dsimg):
        X_blob = torch.from_numpy(self.X).type(torch.float)
        y_blob = torch.from_numpy(self.y).type(torch.LongTensor)

        plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
        plt.savefig(dsimg)
        plt.close()

    def make_circle_dataset_fig(self,dsimg):
        df = pd.DataFrame({"X1": self.X[:, 0], "X2": self.X[:, 1], "label": self.y})

        plt.scatter(df["X1"], df["X2"], c=self.y, cmap=plt.cm.RdYlBu)
        plt.savefig(dsimg)
        plt.close()

    def classifier_train_loop(self, loss_fn, optimizer, epochs):

        self.train_loss = 0
        self.test_loss = 0

        for _ in range(epochs + 1):
            y_logits = self.model(self.X_train).squeeze()

            loss = loss_fn(y_logits, self.y_train)
            self.train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.inference_mode():
                test_logits = self.model(self.X_test).squeeze()

                test_loss = loss_fn(test_logits, self.y_test)
                self.test_loss += test_loss

            self.train_loss = self.train_loss.clone()
            self.test_loss = self.test_loss.clone()
            
            self.train_loss /= epochs
            self.test_loss /= epochs

    def run (self, dsimg, endimg):
        if self.dataset_type == "Make Circles":
            self.X, self.y = make_circles(
                self.samples, noise=self.noise, random_state=self.seed
            )
            self.make_circle_dataset_fig(dsimg)
            loss_fn = nn.BCEWithLogitsLoss()

            self.X = torch.from_numpy(self.X).type(torch.float)
            self.y = torch.from_numpy(self.y).type(torch.float)

        elif self.dataset_type == "Make Blobs":
            self.X, self.y = make_blobs(
                n_samples=self.samples,
                n_features=self.in_features,
                centers=self.classes,
                cluster_std=1.5,
                random_state=self.seed,
            )
            self.make_blob_dataset_fig(dsimg)
            loss_fn = nn.CrossEntropyLoss()

            self.X = torch.from_numpy(self.X).type(torch.float)
            self.y = torch.from_numpy(self.y).type(torch.LongTensor)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed
        )
        if self.activation == "ReLU":
            self.model = ClassifierModelRelu(
                self.in_features, self.out_features, self.hidden_features, self.layer_count
            )
        elif self.activation == "Sigmoid":
            self.model = ClassifierModelSigmoid(
                self.in_features, self.out_features, self.hidden_features, self.layer_count
            )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.classifier_train_loop(loss_fn, optimizer, self.epochs)

        self.plot_decision_boundary(endimg)
        return self.train_loss.item(), self.test_loss.item()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear_stack = nn.Sequential(nn.Linear(1, 1))
                                       
    def forward(self, x):
        return self.linear_stack(x)

class RegressionPipeline:
    def __init__(
        self,
        epochs=None,
        lr=None,
        samples=None,
        test_size=None,
        random_state=None,
        noise=None,
    ):

        self.epochs = epochs
        self.lr = lr
        self.test_size = test_size
        self.random_state = random_state
        self.noise = noise
        self.samples = samples
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predicted = None
        self.train_loss = 0
        self.test_loss = 0

    def generate_data(self):
        X, y = make_regression(
            n_samples=self.samples,
            n_features=1,
            noise=self.noise,
            random_state=self.random_state,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self):
        self.model = LinearRegressionModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            inputs = torch.from_numpy(self.X_train).float()
            labels = torch.from_numpy(self.y_train).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            self.train_loss += loss
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.inference_mode():
                inputs_test = torch.from_numpy(self.X_test).float()
                labels_test = torch.from_numpy(self.y_test).float().view(-1, 1)
                
                output_test = self.model(inputs_test)
                loss_test = criterion(output_test, labels_test)
                self.test_loss += loss_test
        
        self.train_loss = self.train_loss.clone()
        self.test_loss = self.test_loss.clone()
            
        self.train_loss /= self.epochs
        self.test_loss /= self.epochs

    def predict(self):
        inputs = torch.from_numpy(self.X_test).float()
        self.predicted = self.model(inputs).detach().numpy()

    def plot_results(self, dsimg=None, endimg=None, prediction=None):
        plt.scatter(self.X_train, self.y_train, color="blue", label="Training data")
        plt.scatter(self.X_test, self.y_test, color="green", label="Testing data")
        plt.xlabel("X")
        plt.ylabel("y")
        if prediction is not None:
            plt.scatter(self.X_test, self.predicted, color="red", label="Predicted")
            plt.savefig(endimg)
        else:
            plt.savefig(dsimg)
        plt.close()

    def run(self, dsimg, endimg):
        self.generate_data()
        self.plot_results(dsimg=dsimg)
        self.train_model()
        self.predict()
        self.plot_results(endimg=endimg, prediction=self.predicted)
        return self.train_loss.item(), self.test_loss.item()