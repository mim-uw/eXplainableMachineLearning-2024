# Homework 2

The goal of the second homework is to become familiar with SHapley Additive exPlanations (SHAP). 

Calculate these explanations on the model from Homework 1 and gather conclusions from the results.

Focus on the presentation of results; for technical issues, seek support at [Materials towards Homework 2: SHAP with XGBoost & SVM](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw2_shap_with_xgboost_on_titanic.html).

Submit the homework to this directory.

## Deadline 

2023-10-19 EOD

## Task A

For the selected models, prepare a knitr/jupyter notebook based on the following points (you can use models created in Homework 1).
Submit your results on GitHub to the directory `Homeworks/HW2`.

1. Train a tree-based ensemble model on the selected dataset; it can be one of random forest, GBM, CatBoost, XGBoost, LightGBM (various types) etc.
2. Select two observations from the dataset and calculate the model's prediction.
3. Next, for the same observations, calculate the decomposition of predictions, so-called variable attributions, using SHAP from two packages of choice, e.g. for Python: `dalex` and `shap`, for R: `DALEX` and `iml`.
4. Find any two observations in the dataset, such that they have different variables of the highest importance, e.g. age and gender have the highest (absolute) attribution for observation A, but race and class are more important for observation B.
5. (If possible) Select one variable X and find two observations in the dataset such that for one observation, X has a positive attribution, and for the other observation, X has a negative attribution.
6. (How) Do the results differ across the two packages selected in point (3)?
7. (Using one explanation package of choice) Train another model of any class: neural network, linear model, decision tree etc. and find an observation for which SHAP attributions are different between this model and the one trained in point (1).
8. Comment on the results obtained in points (4)-(7)

## Format

The submitted homework should consist of two parts:

1. The 1st part is the key results and comments from points (4)-(7). In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (FIGURES, COMMENTS).**
2. The 2nd part should start with the word "Appendix" or "Załącznik" and should include the reproducible R/Python code used to implement points (1)-(5) & (7).

## Task B

Calculate Shapley values for player A given the following value function

```
v() = 0
v(A) = 20
v(B) = 20
v(C) = 60
v(A,B) = 60
v(A,C) = 70
v(B,C) = 70
v(A,B,C) = 100
```

## **Important note:**

Try to convert the jupyter notebook into an HTML file, e.g. using the following command in bash

```
jupyter nbconvert --to=html --template=classic FILE_NAME.ipynb
```


