#%%
import torch
import pandas as pd
import numpy as np
import torch.nn as nn


import  sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.optim as optim

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%

dataset_path = "imbalanced-benchmarking-set/datasets/MagicTelescope.csv"

df = pd.read_csv(dataset_path)


df = df.drop(df.columns[0], axis=1)

# Y - "TARGET"
# X - all other columns

Y = df["TARGET"]
X = df.drop(columns=["TARGET"])

unique_Y = Y.unique()
map_Y = dict(zip(unique_Y, range(len(unique_Y))))
print(map_Y)

Y = Y.map(map_Y)

number_of_classes = len(unique_Y)
print("Number of classes: ", number_of_classes)

number_of_features = len(X.columns)
print("Number of features: ", number_of_features)

X_df = X.copy()
Y_df = Y.copy()

X = X.to_numpy()
Y = Y.to_numpy()

X_not_normalized = X.copy()
X = sklearn.preprocessing.normalize(X)

print("X shape: ", X.shape)
print("Y shape: ", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_test shape: ", Y_test.shape)

#%%

# Assuming you're doing a classification task
lgb_model = LGBMClassifier(n_estimators=200)
lgb_model.fit(X_train, Y_train)

y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

# %% 1. PVI (LGBM)
def PVI(model,name="Model"):
    result = permutation_importance(
        model, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2
    )

    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_df.columns[sorted_idx])
    ax.set_title(f"Permutation Importances {name}")
    fig.tight_layout()
    plt.show()

PVI(lgb_model,"Large LGBM")


# %% 3.

# LGBM model feature_importances_
lgbm_importance = lgb_model.feature_importances_

def draw_sorted_feature_importance(importance, columns, title):
    sorted_idx = importance.argsort()
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.barh(columns[sorted_idx], importance[sorted_idx])
    plt.show()
    
draw_sorted_feature_importance(lgbm_importance, X_df.columns, "LGBM feature_importances_")




#%%
# SHAP Variable Importance

explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)
shap_importance = np.mean(np.mean(np.abs(shap_values), axis=0),axis=0)

draw_sorted_feature_importance(shap_importance, X_df.columns, "Shap mean Importance")


# %% 2 DNN models
layer_size = 32
layers = []
layers.append(nn.Linear(number_of_features, layer_size))
layers.append(nn.LeakyReLU())
layers.append(nn.Linear(layer_size, layer_size))
layers.append(nn.LeakyReLU())
layers.append(nn.Linear(layer_size, number_of_classes))
layers.append(nn.Softmax(dim=1))
model_weak = nn.Sequential(*layers).to(device)


layer_size = 128
layers = []
layers.append(nn.Linear(number_of_features, layer_size))
layers.append(nn.LeakyReLU())
layers.append(nn.Linear(layer_size, layer_size))
layers.append(nn.BatchNorm1d(layer_size))
layers.append(nn.LeakyReLU())
layers.append(nn.Linear(layer_size, layer_size))
layers.append(nn.BatchNorm1d(layer_size))
layers.append(nn.LeakyReLU())
layers.append(nn.Linear(layer_size, layer_size))
layers.append(nn.BatchNorm1d(layer_size))
layers.append(nn.LeakyReLU())
layers.append(nn.Linear(layer_size, number_of_classes))
layers.append(nn.Softmax(dim=1))
model_stronk = nn.Sequential(*layers).to(device)

#%%


X_train_torch = torch.from_numpy(X_train).float().to(device)
Y_train_torch = torch.from_numpy(Y_train).long().to(device)

dataset = torch.utils.data.TensorDataset(X_train_torch, Y_train_torch)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# %% Feature Importance PVI (2 DNN models)


class TorchModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)

    def fit(self, X, y=None):
        for epoch in tqdm(range(5)):
            epoch_loss = 0
            for batch in dataset_loader:
                self.optimizer.zero_grad()
                X_batch, Y_batch = batch
                Y_pred = self.model(X_batch)
                loss = self.criterion(Y_pred, Y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print("Epoch: ", epoch, " Loss: ", epoch_loss/len(dataset_loader))
        
        X_test_torch = torch.from_numpy(X_test).float().to(device)
        Y_test_torch = torch.from_numpy(Y_test).long().to(device)

        
        Y_pred = self.model(X_test_torch)
        Y_pred = torch.argmax(Y_pred, dim=1)

        print("Accuracy: ", accuracy_score(Y_test_torch.cpu().numpy(), Y_pred.cpu().numpy()))

        return self

    def predict(self, X):
        X_torch = torch.from_numpy(X).float().to(device)
        Y_pred = self.model(X_torch)
        Y_pred = torch.argmax(Y_pred, dim=1)
        return Y_pred.cpu().numpy()


model_wrapper_weak = TorchModelWrapper(model_weak)
model_wrapper_weak.fit(X_train, Y_train)

model_wrapper_stronk = TorchModelWrapper(model_stronk)
model_wrapper_stronk.fit(X_train, Y_train)
    
PVI(model_wrapper_weak,"Weak model")
PVI(model_wrapper_stronk,"Stronk model")    

# %% 
lgb_model = LGBMClassifier(num_leaves=10, max_depth=3, n_estimators=30, random_state=42)
lgb_model.fit(X_train, Y_train)

y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# %% PVI (LGBM)
PVI(lgb_model,"LGBM")
