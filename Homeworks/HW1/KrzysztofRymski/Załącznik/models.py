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

#%% set device
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

# %% Models (My)

# Logistic Regression
model_LR = sklearn.linear_model.LogisticRegression(
    penalty="l2", C=1.0, solver="liblinear", max_iter=1000
)

# Decision Tree
model_DT = sklearn.tree.DecisionTreeClassifier(
    criterion="gini", splitter="best", max_depth=None, min_samples_split=2
)

# Neural Network
internal_size = 128
model_NN_flat = nn.Sequential(
    nn.Linear(number_of_features,internal_size ),
    nn.LeakyReLU(),
    nn.Linear(internal_size, internal_size),
    nn.LeakyReLU(),
    nn.Linear(internal_size, number_of_classes),
    nn.Softmax(dim=1),
)



# Neural Network deep
internal_size_deep = 32
layers = 5

layers_list = []
layers_list.append(nn.Linear(number_of_features, internal_size_deep))
for i in range(layers):
    layers_list.append(nn.LeakyReLU())
    layers_list.append(nn.Linear(internal_size_deep, internal_size_deep))
layers_list.append(nn.LeakyReLU())
layers_list.append(nn.Linear(internal_size_deep, number_of_classes))
layers_list.append(nn.Softmax(dim=1))

model_NN_deep = nn.Sequential(*layers_list)

model_list_sklearn = [model_LR, model_DT]
model_names_sklearn = ["Logistic Regression", "Decision Tree"]
model_list_pytorch = [model_NN_flat, model_NN_deep]
model_names_pytorch = ["Neural Network flat", "Neural Network deep"]

# %% Training

for model in model_list_sklearn:
    model.fit(X_train, Y_train)
    
for model in model_list_pytorch:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long()
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    # allow for early stopping
    patience = 5
    patience_counter = 0
    best_loss = np.inf
    for epoch in tqdm(range(100)):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            X_batch, Y_batch = batch
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
    print("Best loss: ", best_loss)
#%% Evaluation

for model,model_name in zip(model_list_sklearn, model_names_sklearn):
    Y_pred = model.predict(X_test)
    print("Model: ", model_name)
    print("Accuracy: ", accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, Y_pred))
    
dataset_test = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long()
)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False)    

for model,model_name in zip(model_list_pytorch, model_names_pytorch):
    model.eval()
    Y_pred = []
    
    for batch in dataloader_test:
        X_batch, Y_batch = batch
        output = model(X_batch)
        Y_pred.append(torch.argmax(output, dim=1).numpy())
    Y_pred = np.concatenate(Y_pred)
    print("Model: ", model_name)
    print("Accuracy: ", accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, Y_pred))
    
    
# %%

from tabpfn import TabPFNClassifier

classifier = TabPFNClassifier(device=device, N_ensemble_configurations=32)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_not_normalized, Y, test_size=0.33, random_state=42
)

# first 1k samples due to memory limits
X_train = X_train[:1024]
Y_train = Y_train[:1024]

classifier.fit(X_train, Y_train,overwrite_warning=True)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
print("Model: ", "TabPFN")
print('Accuracy', accuracy_score(Y_test, y_eval))
print(classification_report(Y_test, y_eval))
print("Confusion matrix: ")
print(confusion_matrix(Y_test, y_eval))
# %%
