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
#df["age"] = sklearn.preprocessing.normalize(df["age"].to_numpy().reshape(1, -1))[0]
# quantize age by 10
df["age"] = (df["age"] / 10).astype(int)

#df["hours-per-week"] = sklearn.preprocessing.normalize(df["hours-per-week"].to_numpy().reshape(1, -1))[0]

# quantize hours-per-week by 10
df["hours-per-week"] = (df["hours-per-week"] / 10).astype(int)

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
# for column in columns_to_convert:
#     for unique_value in df[column].unique():
#         df[column + "_" + str(unique_value)] = (df[column] == unique_value).astype(int)

#     df = df.drop(columns=[column])
    
df.head()
# %%
# Split data into X and Y


# import aif360
# apply bias mitigation for age gender and race









    
# Y = df["income"]
# X = df.drop(columns=["income"])

# # Convert to numpy
# X = X.to_numpy()
# Y = Y.to_numpy()

# # Convert y to one-hot encoding (2 classes)
# Y = np.eye(2)[Y]

# X_max = np.max(X, axis=0)

# # sum x_max to get total length
# X_total_length = np.sum(X_max)


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



        



# %%
