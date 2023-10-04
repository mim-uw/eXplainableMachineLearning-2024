# Homework 5

The purpose of the fifth homework is to become familiar with Permutation-based Variable Importance (PVI). 

Calculate these explanations on the model from previous Homeworks and gather conclusions from the results.

Focus on the presentation of results; for technical issues, seek support at [Materials towards Homework 5: PVI with XGBoost](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw5_pvi_with_xgboost_on_titanic.html).

Submit the homework to this directory.

## Deadline 

2023-11-16 EOD

## Task

For the selected dataset and models, prepare a knitr/jupyter notebook based on the following points (you can reuse models from previous Homeworks).
Submit your results on GitHub to the directory `Homeworks/HW5`.

0. For the selected data set, train at least one tree-based ensemble model, e.g. random forest, gbdt, xgboost.
1. Calculate Permutation-based Variable Importance for the selected model.
2. Train three more candidate models (different variable transformations, different model architectures, hyperparameters) and compare their rankings of important features using PVI. What are the differences? Why?
3. For the tree-based model from (1), compare PVI with: 
    - A) the traditional feature importance measures for trees: Gini impurity etc.; what is implemented in a given library: see e.g. the `feature_importances_` attribute in `xgboost` and `sklearn`.
    - B) [in Python] SHAP variable importance based on the TreeSHAP algorithm available in the `shap` package. 
    
    What are the differences? Why?
4. ! COMMENT on the results obtained in (1)-(3)



## **Important note:**

Try to convert the jupyter notebook into an HTML file, e.g. using the following command in bash

```
jupyter nbconvert --to=html --template=classic FILE_NAME.ipynb
```

The submitted homework should consist of two parts:

1. The 1st part is the key results and comments from points (2)-(5). In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (FIGURES, COMMENTS).**
2. The 2nd part should start with the word "Appendix" or "Załącznik" and should include the reproducible R/Python code used to implement points (1)-(5).

Such division: 1. will make this homework more readable, and 2. will develop good habits related to reporting.
