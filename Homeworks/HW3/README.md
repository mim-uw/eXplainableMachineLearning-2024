# Homework 3

The purpose of the third homework is to become familiar with Local Interpretable Model-agnostic Explanations (LIME). 

Calculate these explanations on the model from previous Homeworks and gather conclusions from the results.

Focus on the presentation of results; for technical issues, seek support at [Materials towards Homework 3: LIME with XGBoost & SVM](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw3_lime_with_xgboost_on_titanic.html).

Submit the homework to this directory.

## Deadline 

2023-10-26 EOD

## Task

For the selected dataset and models, prepare a knitr/jupyter notebook based on the following points (you can reuse models from previous Homeworks).
Submit your results on GitHub to the directory `Homeworks/HW3`.

1. Calculate the predictions for some selected observations
2. Then, calculate the decomposition of these predictions with `LIME` using the package of choice, e.g. in Python: `lime`, `dalex`, in R: `iml`, `localModel`.
3. Compare LIME for various observations in the dataset. How stable are these explanations? 
4. Compare LIME with the explanations obtained using SHAP. What are the main differences between them?
5. Compare LIME between at least two different models. Are there any systematic differences across many observations?
6. Comment on the results obtained in (3), (4) and (5)


## **Important note:**

Try to convert the jupyter notebook into an HTML file, e.g. using the following command in bash

```
jupyter nbconvert --to=html --template=classic FILE_NAME.ipynb
```

The submitted homework should consist of two parts:

1. The 1st part is the key results and comments from points (3)-(5). In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (FIGURES, COMMENTS).**
2. The 2nd part should start with the word "Appendix" or "Załącznik" and should include the reproducible R/Python code used to implement points (1)-(5).

Such division: 1. will make this homework more readable, and 2. will develop good habits related to reporting.
