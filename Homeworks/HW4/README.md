# Homework 4

The purpose of the fourth homework is to become familiar with Ceteris Paribus (CP) and Partial Dependence profiles (PDP), and its variants like Accumulated Local Effects (ALE). 

Calculate these explanations on the model from previous Homeworks and gather conclusions from the results.

Focus on the presentation of results; for technical issues, seek support at [Materials towards Homework 4: CP and PDP with XGBoost](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw4_cp_and_pdp_with_xgboost_on_titanic.html).

Submit the homework to this directory.

## Deadline 

2023-11-09 EOD

## Task 1

Consider a following model:

f(x1, x2) = (x1 + x2)^2

Assume that x1, x2 ~ U[-1,1] and x1=x2 (full dependency)

Calculate PD profile for variable x1 in this model.

Extra task if you do not fear conditional expected values: Calculate ME and ALE profiles for variable x1 in this model.


## Task 2

For the selected dataset and models, prepare a knitr/jupyter notebook based on the following points (you can reuse models from previous Homeworks).
Submit your results on GitHub to the directory `Homeworks/HW4`.


0. For the selected data set, train at least one tree-based ensemble model, e.g. random forest, gbdt, xgboost.
1. Calculate the predictions for some selected observations.
2. Then, calculate the what-if explanations of these predictions using Ceteris Paribus profiles (also called What-if plots), e.g. in Python: `AIX360`, `Alibi` `dalex`, `PDPbox`; in R: `pdp`, `DALEX`, `ALEplot`. **implement CP yourself for a potential bonus point*
3. Find two observations in the data set, such that they have different CP profiles. For example, model predictions are increasing with `age` for one observation and decreasing with `age` for another one. NOTE that you will need to have a model with interactions to observe such differences.
4. Compare CP, which is a local explanation, with PDP, which is a global explanation. **implement PDP yourself for a potential bonus point*
5. Compare PDP between between at least two different models.
6. ! COMMENT on the results obtained in (2)-(5)


## **Important note:**

Try to convert the jupyter notebook into an HTML file, e.g. using the following command in bash

```
jupyter nbconvert --to=html --template=classic FILE_NAME.ipynb
```

The submitted homework should consist of two parts:

1. The 1st part is the key results and comments from points (2)-(5). In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (FIGURES, COMMENTS).**
2. The 2nd part should start with the word "Appendix" or "Załącznik" and should include the reproducible R/Python code used to implement points (1)-(5).

Such division: 1. will make this homework more readable, and 2. will develop good habits related to reporting.
