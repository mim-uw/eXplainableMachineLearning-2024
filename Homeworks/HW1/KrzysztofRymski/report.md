# Homework 1
As we can see that more complex models to implement, the better results we get. TabPFN is the strongest one with 86% accuracy given that it only seen 1k of samples compared to 12k for other models. The dataset used was MagicTelescope but one can use also other datasets from imbalanced-benchmarking-set repo and the code should work just fine.

Models used were:
- Logistic Regression
- Decision Tree
- Neural Network flat (1 hidden layer 128 neurons) 
- Neural Network deep (5 hidden layers 32 neurons each)
- TabPFN

# Results

## Model:  Logistic Regression
    Accuracy:  0.7270989326111199
                precision    recall  f1-score   support

            0       0.73      0.92      0.81      4071
            1       0.71      0.38      0.49      2206

    accuracy                            0.73      6277
    macro avg       0.72      0.65      0.65      6277
    weighted avg    0.72      0.73      0.70      6277

    Confusion matrix: 
    [[3728  343]
    [1370  836]]

## Model:  Decision Tree

    Accuracy:  0.781742870798152
                precision    recall  f1-score   support

            0       0.83      0.84      0.83      4071
            1       0.70      0.67      0.68      2206

    accuracy                            0.78      6277
    macro avg       0.76      0.76      0.76      6277
    weighted avg    0.78      0.78      0.78      6277

    Confusion matrix: 
    [[3418  653]
    [ 717 1489]]

## Model:  Neural Network flat
    Accuracy:  0.827146726143062
                precision    recall  f1-score   support

            0       0.82      0.94      0.88      4071
            1       0.84      0.63      0.72      2206

    accuracy                            0.83      6277
    macro avg       0.83      0.78      0.80      6277
    weighted avg    0.83      0.83      0.82      6277

    Confusion matrix: 
    [[3807  264]
    [ 821 1385]]

## Model:  Neural Network deep
    Accuracy:  0.822685996495141
                precision    recall  f1-score   support

            0       0.82      0.92      0.87      4071
            1       0.82      0.64      0.72      2206

    accuracy                            0.82      6277
    macro avg       0.82      0.78      0.79      6277
    weighted avg    0.82      0.82      0.82      6277

    Confusion matrix: 
    [[3755  316]
    [ 797 1409]]


## Model:  TabPFN  25.82 M

    Accuracy 0.8658594870160905
                precision    recall  f1-score   support

            0       0.87      0.93      0.90      4071
            1       0.86      0.74      0.80      2206

    accuracy                            0.87      6277
    macro avg       0.86      0.84      0.85      6277
    weighted avg    0.87      0.87      0.86      6277

    Confusion matrix: 
    [[3794  277]
    [ 565 1641]]
