# Homework 6

The purpose of the sixth homework is to learn about the method of calculating fairness statistics.

Calculate these statistics for a dataset of 'credit scoring' or 'adult income'.

Focus on the presentation of results; for technical issues, seek support at [Materials towards Homework 6: Fairness with XGBoost](https://mim-uw.github.io/eXplainableMachineLearning-2023/hw6_fairness_with_xgboost_on_titanic.html).

Submit the homework to this directory.

## Deadline 

2022-11-23 EOD

## Task 1
We have two populations Blue (privileged) and Red (unprivileged), with the Blue population being 9 times larger than the Red population.

Individuals from both populations are requesting to attend XAI training to improve competency in this important area. Number of places is limited. The administrators of the training have decided to give priority to enrolling individuals who may need this training in the future, although unfortunately it is difficult to predict who will benefit.

The decision rule adopted:
1. In the Red group, half of the people will find the skills useful in future and half will not. Administrators randomly allocate 50% of people to training.
2. in the Blue group, 80% of people will find the training useful in future and 20% will not, although of course it is not known who will find it useful. The administrators have built a predictive model based on user behaviour in predicting for whom it will be useful and whom will not. The model has the following performance:


| Blue                     	| Will use XAI 	| Will not use XAI 	| Total 	|
|--------------------------	|--------------	|------------------	|-------	|
| Enrolled in training     	| 60           	| 5               	| 65    	|
| not enrolled in training 	| 20            	| 15               	| 35    	|
| Total                    	| 80           	| 20               	| 100   	|


Task: Calculate the Demographic parity, equal opportunity and predictive rate parity coefficients for this decision rule.

Starred task: How can this decision rule be changed to improve its fairness?


## Task 2

For this homework, train models on one of the following datasets:

- credit scoring https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset (https://www.kaggle.com/competitions/GiveMeSomeCredit/)
- adult income https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

Prepare a knitr/jupiter notebook with the following points.
Submit your results on GitHub to the directory `Homeworks/HW6`.

1. Train a model for the selected dataset.
2. For the selected protected attribute (age, gender, race) calculate the following fairness coefficients: Statistical parity, Equal opportunity, Predictive parity.
3. Train another model (different hyperparameters, feature transformations etc.) and see how the coefficients Statistical parity, Equal opportunity, Predictive parity behave for it.
4. Apply the selected bias mitigation technique on the first model. Check how Statistical parity, Equal opportunity, Predictive parity coefficients behave after this correction.
5. Compare the quality (performance) of the three models with their fairness coefficients. Is there any correlation?
6. ! COMMENT on the results obtained in (2)-(5)



## **Important note:**

Try to convert the jupyter notebook into an HTML file, e.g. using the following command in bash

```
jupyter nbconvert --to=html --template=classic FILE_NAME.ipynb
```

The submitted homework should consist of two parts:

1. The 1st part is the key results and comments from points (2)-(5). In this part **PLEASE DO NOT SHOW ANY R/PYTHON CODES, ONLY RESULTS (FIGURES, COMMENTS).**
2. The 2nd part should start with the word "Appendix" or "Załącznik" and should include the reproducible R/Python code used to implement points (1)-(5).

