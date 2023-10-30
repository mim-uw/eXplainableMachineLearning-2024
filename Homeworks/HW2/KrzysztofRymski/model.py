#%%
import pandas as pd
import numpy as np
import torch.nn as nn
import sklearn.preprocessing
import torch
#age,workclass,fnlwgt,education,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country,income
#25,Private,226802,11th,7,Never-married,Machine-op-inspct,Own-child,Black,Male,0,0,40,United-States,<=50K
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%

df = pd.read_csv("adult.csv")

df.head()

# Preprocessing

# Drop fnlwgt, education, capital-gain, capital-loss, native-country
# This data might be very inbalanced and might generate overfitting
df = df.drop(columns=["fnlwgt", "education", "capital-gain", "capital-loss", "native-country"])

df.head()
# %%

# Age and hours-per-week are continuous variables
# Let's normalize them and convert them to float
# For the rest we will use one-hot encoding then we will flatten the data

# Normalize age and hours-per-week
df["age"] = sklearn.preprocessing.normalize(df["age"].to_numpy().reshape(1, -1))[0]
df["hours-per-week"] = sklearn.preprocessing.normalize(df["hours-per-week"].to_numpy().reshape(1, -1))[0]

# Integer encoding
columns_to_convert = ["workclass", "marital-status", "occupation", "relationship", "race", "gender", "income"]

for column in columns_to_convert:
    df[column] = df[column].astype('category').cat.codes

columns_to_convert.remove("income")

df.head()
# %%

# One-hot encoding and flattening
# total length of one hot

total_length = 2
for column in columns_to_convert:
    total_length += len(df[column].unique())

print("Total length: ", total_length)

# Itterate over rows and create column with one-hot encoding
# Then drop the original column
for column in columns_to_convert:
    for unique_value in df[column].unique():
        df[column + "_" + str(unique_value)] = (df[column] == unique_value).astype(int)

    df = df.drop(columns=[column])
    
df.head()
# %%
# Split data into X and Y
Y = df["income"]
X = df.drop(columns=["income"])

# Convert to numpy
X = X.to_numpy()
Y = Y.to_numpy()

# Convert y to one-hot encoding (2 classes)
Y = np.eye(2)[Y]

# Convert to float
X = X.astype(float)
Y = Y.astype(float)


# %%

# Split data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

X_train = torch.Tensor(X_train).to(device)
Y_train = torch.Tensor(Y_train).to(device)





# %%

# Create model using pytorch 5 layers parametrized 

def create_model(num_layers, layer_size):
    layers = []
    layers.append(nn.Linear(X_train.shape[1], layer_size))
    for i in range(num_layers):
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(layer_size, layer_size))
    layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(layer_size, Y_train.shape[1]))
    layers.append(nn.Softmax(dim=1))

    model = nn.Sequential(*layers)
    return model


hidden_layer_size = 64

model_1 = create_model(3, hidden_layer_size)
model_2 = create_model(5, hidden_layer_size*2)
model_1_fair = create_model(3, hidden_layer_size)

models = [model_1, model_2, model_1_fair]
for model in models:
    model.to(device)
    

# %%

# Create loss function and optimizer
import torch.optim as optim

loss_function = nn.MSELoss()
optimizers = []
optimizers.append(optim.Adam(model_1.parameters(), lr=0.001))
optimizers.append(optim.Adam(model_2.parameters(), lr=0.001))
optimizers.append(optim.Adam(model_1_fair.parameters(), lr=0.001))

# %%
# Train the models 1 and 2
import matplotlib.pyplot as plt
from tqdm import tqdm
epochs = 1000

losses = [[] for i in range(len(optimizers))]




# train models 1 and 2 only
dataset = torch.utils.data.TensorDataset(
    torch.Tensor(X_train), torch.Tensor(Y_train)
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for model_id in range(2):
    early_stop = 0
    best_loss = np.inf
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for batch in dataloader:
            optimizers[model_id].zero_grad()
            X_batch, Y_batch = batch
            output = models[model_id](X_batch)
            loss = loss_function(output, Y_batch)
            loss.backward()
            optimizers[model_id].step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        losses[model_id].append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > 10:
            break


# %%
# Plot losses
plt.figure(figsize=(10, 10))
for i, loss in enumerate(losses):
    plt.plot(loss, label="Model " + str(i))
    

plt.legend()

# %%
# Evaluate models and compute  Statistical parity, Equal opportunity, Predictive parity coefficients behave after this correction.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# X_test to tensor
X_test = torch.Tensor(X_test).to(device)
Y_test = torch.Tensor(Y_test).to(device)

def evaluate_model(model, X_test, Y_test):
    Y_pred = model(X_test)
    Y_pred = torch.argmax(Y_pred, dim=1)
    Y_test = torch.argmax(Y_test, dim=1)
    accuracy = accuracy_score(Y_test.cpu(), Y_pred.cpu())
    precision = precision_score(Y_test.cpu(), Y_pred.cpu())
    recall = recall_score(Y_test.cpu(), Y_pred.cpu())
    f1 = f1_score(Y_test.cpu(), Y_pred.cpu())
    return accuracy, precision, recall, f1

for model_id, model in enumerate([model_1, model_2]):
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, Y_test)
    print("Model " + str(model_id) + " accuracy: ", accuracy)
    print("Model " + str(model_id) + " precision: ", precision)
    print("Model " + str(model_id) + " recall: ", recall)
    print("Model " + str(model_id) + " f1: ", f1)
    print("")
    
#%%
# Compute Statistical parity, Equal opportunity, Predictive parity coefficients

def compute_coefficients(model, X_test, Y_test):
    Y_pred = model(X_test)
    Y_pred = torch.argmax(Y_pred, dim=1)
    Y_test = torch.argmax(Y_test, dim=1)
    Y_pred = Y_pred.cpu().numpy()
    Y_test = Y_test.cpu().numpy()
    # round to 0 or 1 0.5 threshold
    Y_pred = np.round(Y_pred)
    Y_test = np.round(Y_test)
    
    Y_pred = Y_pred.astype(int)
    Y_test = Y_test.astype(int)
    # Statistical parity
    # P(Y=1) - P(Y=0)
    p_y_1 = np.sum(Y_test == 1) / len(Y_test)
    p_y_0 = np.sum(Y_test == 0) / len(Y_test)
    statistical_parity = p_y_1 - p_y_0
    # Equal opportunity
    # P(Y=1|Y_hat=1) - P(Y=1|Y_hat=0)
    p_y_1_y_hat_1 = np.sum((Y_test == 1) & (Y_pred == 1)) / np.sum(Y_pred == 1)
    p_y_1_y_hat_0 = np.sum((Y_test == 1) & (Y_pred == 0)) / np.sum(Y_pred == 0)
    equal_opportunity = p_y_1_y_hat_1 - p_y_1_y_hat_0
    # Predictive parity
    # P(Y_hat=1|Y=1) - P(Y_hat=1|Y=0)
    p_y_hat_1_y_1 = np.sum((Y_test == 1) & (Y_pred == 1)) / np.sum(Y_test == 1)
    p_y_hat_1_y_0 = np.sum((Y_test == 0) & (Y_pred == 1)) / np.sum(Y_test == 0)
    predictive_parity = p_y_hat_1_y_1 - p_y_hat_1_y_0
    return statistical_parity, equal_opportunity, predictive_parity

for model_id, model in enumerate([model_1, model_2]):
    statistical_parity, equal_opportunity, predictive_parity = compute_coefficients(model, X_test, Y_test)
    print("Model " + str(model_id) + " statistical parity: ", statistical_parity)
    print("Model " + str(model_id) + " equal opportunity: ", equal_opportunity)
    print("Model " + str(model_id) + " predictive parity: ", predictive_parity)
    print("")


# %%

# Train model 1 fair

