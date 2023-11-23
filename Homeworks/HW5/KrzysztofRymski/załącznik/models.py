#%%
import torch
import pandas as pd
import numpy as np
import torch.nn as nn


import  sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

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
import lightgbm as lgb


lgb_train = lgb.Dataset(X_train, Y_train)

lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

params = {
    "objective": "multiclass",
    "num_class": number_of_classes,
    "metric": "multi_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
}


print("Starting training...")


lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=lgb_eval,
)


y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)


accuracy = accuracy_score(Y_test, np.argmax(y_pred, axis=1))

print("Accuracy: %.2f%%" % (accuracy * 100.0))

# %%

import dalex

#%%
#Then, calculate the what-if explanations of these predictions using Ceteris Paribus profiles (also called What-if plots),
import dalex as dx
#%% [markdown]
# Task 2,3 - Ceteris Paribus profiles
# I used dalex library to calculate Ceteris Paribus profiles
# Results are in the graphs below
# To find a prediction with different CP profiles I simply have taken the very next observation from the test set
# Then I would like to focus your attention of fWidth, as it has almost the exact opposite effect on the prediction
# At the same time, fLength produced quite similar graphs for both observations


#%%
sample_id = 0

feature_names = X_df.columns


X_train_df = pd.DataFrame(X_train, columns=feature_names)

pf_0 = lambda m,d: m.predict(d)[:,0]

explainer_dalex = dx.Explainer(lgb_model, X_train_df, Y_train, predict_function=pf_0, label="MagicTelescope")

cp = explainer_dalex.predict_profile(X_test[sample_id], variables=feature_names.to_list())
cp.plot()

# %%
cp2 = explainer_dalex.predict_profile(X_test[sample_id+1], variables=feature_names.to_list())
cp2.plot()

#%% [markdown]
# Task 4 - Partial Dependence Plots
# Here we can see a cute result where fWdith gives us very similar plot to what we have seen in CP profiles for the first sample

#%%
import matplotlib.pyplot as plt
import numpy as np

def calculate_partial_dependence(model, feature_idx, X, grid):
    X_copy = X.copy()
    partial_dependences = []
    
    for val in grid:
        X_copy[:, feature_idx] = val
        predictions = model.predict(X_copy)
        pdp = np.mean(predictions, axis=0)
        partial_dependences.append(pdp)
    
    return partial_dependences

def plot_partial_dependence(feature_name, grid, pdp):
    plt.figure(figsize=(10, 6))
    plt.plot(grid, pdp)
    plt.xlabel(feature_name)
    plt.ylabel('Partial dependence')
    plt.show()

for feature_idx in range(number_of_features):

    grid = np.linspace(np.min(X[:, feature_idx]), np.max(X[:, feature_idx]), num=100)
    pdp = calculate_partial_dependence(lgb_model, feature_idx, X, grid)
    plot_partial_dependence(feature_names[feature_idx], grid, pdp)
# %%
layer_size = 32
layers = []
layers.append(nn.Linear(number_of_features, layer_size))
#layers.append(nn.BatchNorm1d(layer_size)) # Hi uncoment me to see the difference
layers.append(nn.ReLU())
# I nerfed my initial model, as it was overfitting
layers.append(nn.Linear(layer_size, layer_size))
#layers.append(nn.BatchNorm1d(layer_size))
layers.append(nn.ReLU())
# layers.append(nn.Linear(300, 200))
# layers.append(nn.BatchNorm1d(200))
# layers.append(nn.ReLU())
# layers.append(nn.Linear(100, 100))
# layers.append(nn.BatchNorm1d(100))
# layers.append(nn.ReLU())
layers.append(nn.Linear(layer_size, number_of_classes))
layers.append(nn.Softmax(dim=1))
model = nn.Sequential(*layers).to(device)

#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_torch = torch.from_numpy(X_train).float().to(device)
Y_train_torch = torch.from_numpy(Y_train).long().to(device)

dataset = torch.utils.data.TensorDataset(X_train_torch, Y_train_torch)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in tqdm(range(5)):
    epoch_loss = 0
    for batch in dataset_loader:
        optimizer.zero_grad()
        X_batch, Y_batch = batch
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Epoch: ", epoch, " Loss: ", epoch_loss/len(dataset_loader))
    
# %%
X_test_torch = torch.from_numpy(X_test).float().to(device)
Y_test_torch = torch.from_numpy(Y_test).long().to(device)

Y_pred = model(X_test_torch)
Y_pred = torch.argmax(Y_pred, dim=1)

print("Accuracy: ", accuracy_score(Y_test_torch.cpu().numpy(), Y_pred.cpu().numpy()))
# %% [markdown]
# Calculate PDP for other model
# I wanted to again compare against dnn model, so I constructed a simple one in pytorch (2 hidden layer)
# Now I think here happend super interestign thing
# As I though I messed up something, as I was getting straigth parrarel lines for all features
# After a bit of investigation I found that culprit was batch normalization, and general size of the model, ito show you somehow intresting I performed effectively a lobotomy on a model, tho It still has 80% accuracy (bigger one had 83%)
# I will show you results with and without batch normalization, and on small model to make it intresting (I think you can imagine graph 2 pararell lines)
# I think this is because DNN models effectively create new features, and PDP method works kind of like a disruption of one of the features, but with 12 other ones to rely on model dosen't care about it
# To test this theory I increased range I'm scanning over in PDP to one that is outside range observed in the dataset, and as you can see PDP works, but model is very robust in typical range of values
# Bellow is polt with normal range of values, and one with extended range
#%%
class TorchModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        X_torch = torch.from_numpy(X).float().to(device)
        Y_pred = self.model(X_torch)
        return Y_pred.cpu().detach().numpy()
    
model_wraped = TorchModelWrapper(model)

feature_idx = 1
grid = np.linspace(np.min(X[:, feature_idx]),np.max(X[:, feature_idx]), num=1000)
pdp = calculate_partial_dependence(model_wraped, feature_idx, X_test, grid)
plot_partial_dependence(feature_names[feature_idx], grid, pdp)


# %% [markdown]
# Plot with extended range
#%%
grid = np.linspace(np.min(X[:, feature_idx])-1,1+ np.max(X[:, feature_idx]), num=1000)
pdp = calculate_partial_dependence(model_wraped, feature_idx, X_test, grid)
plot_partial_dependence(feature_names[feature_idx], grid, pdp)


# %%
