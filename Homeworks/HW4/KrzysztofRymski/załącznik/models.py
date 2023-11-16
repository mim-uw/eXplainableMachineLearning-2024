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
    early_stopping_rounds=10,
)


y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)


accuracy = accuracy_score(Y_test, np.argmax(y_pred, axis=1))

print("Accuracy: %.2f%%" % (accuracy * 100.0))

# %%

import shap
import dalex as dx
import shap.plots
#%% [markdown]
# TASK 2,3 - DALEX
#
# Compare LIME for various observations in the dataset. How stable are these explanations? 
#
# I don't think those observations are stable, because they are quite different for each observation. There are some patterns, like discovered in previous homeworks, for example fWidth is often one of the most important features. Buth when looking at individual observations, it's hard to find any patterns.


#%%
sample_id = 0

feature_names = X_df.columns


X_train_df = pd.DataFrame(X_train, columns=feature_names)

pf_0 = lambda m,d: m.predict(d)[:,0]

explainer_dalex = dx.Explainer(lgb_model, X_train_df, Y_train, predict_function=pf_0, label="MagicTelescope")
shap_values_dalex = explainer_dalex.predict_parts(X_test[sample_id, :])

shap_values_dalex.plot(show=True)

shap_values_dalex_2 = explainer_dalex.predict_parts(X_test[sample_id+1, :])
shap_values_dalex_2.plot(show=True)
#%% [markdown]
# TASK 2,3 - SHAP
# Same this apply for SHAP

# %%

explainer_shap = shap.TreeExplainer(lgb_model,X_train_df)
shap_values = explainer_shap(X_test[sample_id:sample_id+1, :])
shap.plots.waterfall(shap_values[0], max_display=20)

shap_values_22 = explainer_shap(X_test[sample_id+1:sample_id+2, :])
shap.plots.waterfall(shap_values_22[0], max_display=20)

# %% [markdown]
# Task 4,5 - SHAP later DALEX
#
# I think this task 4 was very simple and shows the instablity found in task 3
#
# For task five I decided to find all pairs of observations that have different most important feature. I found over 96 milons of such pairs, and because is quite funny how many there are, I deciced to print between which pairs they are occuring.
#
# Dalex is much slower so I only printed first such pair

# %%
shap_values_1 = shap_values

def most_important_feture(values):
    most_important = np.argmax(np.abs(values.values))
    return most_important

ct =0
for ct in tqdm(range(len(X_test))):
    shap_values_2 = explainer_shap(X_test[ct:ct+1, :])
    if most_important_feture(shap_values_1) != most_important_feture(shap_values_2):
        print("Observation 1: ",sample_id, " most important feature: ", X_train_df.columns[most_important_feture(shap_values_1)])
        shap.plots.waterfall(shap_values[0], max_display=20)
        print("Observation 2: ",ct, " most important feature: ", X_train_df.columns[most_important_feture(shap_values_2)])
        shap.plots.waterfall(shap_values_2[0], max_display=20)
        break
# %%
observed_positive = {}
observed_negative = {}

for ct in tqdm(range(len(X_test))):
    shap_values_tmp = explainer_shap(X_test[ct:ct+1, :])
    for i in range(len(X_train_df.columns)):
        if shap_values_tmp.values[0][i] > 0:
            if X_train_df.columns[i] in observed_positive:
                observed_positive[X_train_df.columns[i]].append(ct)
            else:
                observed_positive[X_train_df.columns[i]] = [ct]
        else:
            if X_train_df.columns[i] in observed_negative:
                observed_negative[X_train_df.columns[i]].append(ct)
            else:
                observed_negative[X_train_df.columns[i]] = [ct]

#%%
total_pairs = 0
for key in observed_positive:
    if key in observed_negative:
        print("Feature: ", key)
        print("Positive: ", observed_positive[key])
        print("Negative: ", observed_negative[key])
        total_pairs += len(observed_positive[key]) * len(observed_negative[key])
print("Total pairs: ", total_pairs)
# %% [markdown]
# Task 4,5 - DALEX
# text is attached to SHAP section
#%%
import copy


exp1 = explainer_dalex.predict_parts(X_test[0, :])
exp1_copy = copy.deepcopy(exp1)
exp1.result = exp1.result[1:-1]
exp1.result["abs_contribution"] = np.abs(exp1.result["contribution"])
exp1.result = exp1.result.sort_values(by=["abs_contribution"], ascending=False)
exp1.result = exp1.result.reset_index(drop=True)
#%%
for ct in tqdm(range(len(X_test))):
    exp = explainer_dalex.predict_parts(X_test[ct, :])
    exp_copy = copy.deepcopy(exp)
    exp.result = exp.result[1:-1]
    exp.result["abs_contribution"] = np.abs(exp.result["contribution"])
    exp.result = exp.result.sort_values(by=["abs_contribution"], ascending=False)
    exp.result = exp.result.reset_index(drop=True)
    if exp1.result['variable_name'][0] != exp.result['variable_name'][0]:
        print("Observation 1: ", 0, " most important feature: ", exp1.result['variable_name'][0])
        exp1_copy.plot(max_vars=20)
        print("Observation 2: ", ct, " most important feature: ", exp.result['variable_name'][0])
        exp_copy.plot(max_vars=20)
        break

#%%
def feature_with_positive_and_negative_impact():
    observed_positive = {}
    observed_negative = {}
    for ct in tqdm(range(len(X_test))):
        exp = explainer_dalex.predict_parts(X_test[ct, :])
        for i in range(len(exp.result)):
            if exp.result['contribution'][i] > 0:
                if exp.result['variable_name'][i] in observed_positive:
                    observed_positive[exp.result['variable_name'][i]].append((exp,ct))
                else:
                    observed_positive[exp.result['variable_name'][i]] = [(exp,ct)]
            else:
                if exp.result['variable_name'][i] in observed_negative:
                    observed_negative[exp.result['variable_name'][i]].append((exp,ct))
                else:
                    observed_negative[exp.result['variable_name'][i]] = [(exp,ct)]

        for key in observed_positive:
            if key in observed_negative:
                print("Feature with both impacts: ", key)
                print("Observation with positive impact: ", observed_positive[key][0][1])
                print("Observation with negative impact: ", observed_negative[key][0][1])
                observed_positive[key][0][0].plot(max_vars=20)
                observed_negative[key][0][0].plot(max_vars=20)
                return
feature_with_positive_and_negative_impact()
print("Total pairs: ", "Too long to calculate")

# %% [markdown]
# Taks 6,7
# Task-6 I think both of those packages are fairly inconsisten with theirs explantations, and I don't think I would trust them more than on a surface level for feature elimination. I also think that Tree based models are inheretly harded to explain, as linar combination of features, as by their nature they are made out of many if statements, and sub linear models.
 
# Taks-7 Not much to comment, I just trained weaker model it had roughly 2% lower accuracy, but the explanations were similar.
#%%
params = {
    "objective": "multiclass",
    "num_class": number_of_classes,
    "metric": "multi_logloss",
    "num_leaves": 5,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
}


print("Starting training model 2...")

lgb_model2 = lgb.train(
    params,
    lgb_train,
    num_boost_round=50,
    valid_sets=lgb_eval,
)

y_pred = lgb_model2.predict(X_test, num_iteration=lgb_model.best_iteration)

accuracy = accuracy_score(Y_test, np.argmax(y_pred, axis=1))

print("Accuracy: %.2f%%" % (accuracy * 100.0))
#%%

explainer_shap2 = shap.TreeExplainer(lgb_model2,X_train_df)

for ct in tqdm(range(len(X_test))):
    shap_values1_tmp = explainer_shap(X_test[ct:ct+1, :])
    shap_values2_tmp = explainer_shap2(X_test[ct:ct+1, :])
    
    if most_important_feture(shap_values1_tmp) != most_important_feture(shap_values2_tmp):
        print("Observation 1: ",ct, " most important feature: ", X_train_df.columns[most_important_feture(shap_values1_tmp)])
        shap.plots.waterfall(shap_values1_tmp[0], max_display=20)
        print("Observation 2: ",ct, " most important feature: ", X_train_df.columns[most_important_feture(shap_values2_tmp)])
        shap.plots.waterfall(shap_values2_tmp[0], max_display=20)
        break

