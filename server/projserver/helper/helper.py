import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
from torch import nn


class ClassifierModelRelu(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, X):
        return self.linear_layer_stack(X)


class ClassifierModelSigmoid(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, X):
        return self.linear_layer_stack(X)


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
        features=None,
        classes=None,
        activation=None,
    ):

        self.epochs = epochs
        self.dataset_type = dataset_type
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.classes = classes
        self.samples = samples
        self.features = features
        self.seed = seed
        self.noise = noise
        self.model = None
        self.X_tensor = None
        self.y_tensor = None
        self.X = None
        self.y = None
        self.lr = lr
        self.test_size = test_size
        self.train_loss = 0
        self.test_loss = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.activation = activation

    def plot_decision_boundary(self):
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
        plt.savefig("img/endfig_cls.png")
        plt.close()

    def make_blob_dataset_fig(self):
        X_blob = torch.from_numpy(self.X).type(torch.float)
        y_blob = torch.from_numpy(self.y).type(torch.LongTensor)

        plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
        plt.savefig("img/ds_make_blobs.png")
        plt.close()

    def make_circle_dataset_fig(self):
        df = pd.DataFrame({"X1": self.X[:, 0], "X2": self.X[:, 1], "label": self.y})

        plt.scatter(df["X1"], df["X2"], c=self.y, cmap=plt.cm.RdYlBu)
        plt.savefig("ds_make_circles.png")
        plt.close()

    def classifier_train_loop(self, loss_fn, optimizer, epochs):

        self.train_loss = 0
        self.test_loss = 0

        for _ in range(epochs + 1):
            y_logits = self.model(self.X_train).squeeze()
            # y_pred = torch.round(torch.sigmoid(y_logits))

            loss = loss_fn(y_logits, self.y_train)
            self.train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.inference_mode():
                test_logits = self.model(self.X_test).squeeze()
                # test_pred = torch.round(torch.sigmoid(test_logits))

                test_loss = loss_fn(test_logits, self.y_test)
                self.test_loss += test_loss

            self.train_loss = self.train_loss / epochs
            self.test_loss = self.test_loss / epochs

    def classifier_main(self):
        if self.dataset_type == "Make Circles":
            self.X, self.y = make_circles(
                self.samples, noise=self.noise, random_state=self.seed
            )
            self.make_circle_dataset_fig()
            loss_fn = nn.BCEWithLogitsLoss()

            self.X = torch.from_numpy(self.X).type(torch.float)
            self.y = torch.from_numpy(self.y).type(torch.float)

        elif self.dataset_type == "Make Blobs":
            self.X, self.y = make_blobs(
                n_samples=self.samples,
                n_features=self.features,
                centers=self.classes,
                cluster_std=1.5,
                random_state=self.seed,
            )
            self.make_blob_dataset_fig()
            loss_fn = nn.CrossEntropyLoss()

            self.X = torch.from_numpy(self.X).type(torch.float)
            self.y = torch.from_numpy(self.y).type(torch.LongTensor)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed
        )
        if self.activation == "ReLU":
            self.model = ClassifierModelRelu(
                self.in_features, self.out_features, self.hidden_features
            )
        else:
            self.model = ClassifierModelSigmoid(
                self.in_features, self.out_features, self.hidden_features
            )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.classifier_train_loop(loss_fn, optimizer, self.epochs)

        self.plot_decision_boundary()


# ClassifierPipeline(18000, 'Make Circles', 0.2, 0.01, 2, 10, 1, 1000, 42, 0.03, activation='ReLU').classifier_main()
ClassifierPipeline(
    samples=1000,
    classes=4,
    features=2,
    seed=42,
    dataset_type="Make Blobs",
    epochs=15000,
    lr=0.1,
    hidden_features=5,
    in_features=2,
    out_features=4,
    activation="ReLU",
).classifier_main()
