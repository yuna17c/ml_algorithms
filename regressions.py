import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

X_test_A = np.loadtxt(open("datasets/X_test_A.csv", "rb"), delimiter=",")
X_test_B = np.loadtxt(open("datasets/X_test_B.csv", "rb"), delimiter=",")
X_test_C = np.loadtxt(open("datasets/X_test_C.csv", "rb"), delimiter=",")
X_train_A = np.loadtxt(open("datasets/X_train_A.csv", "rb"), delimiter=",")
X_train_B = np.loadtxt(open("datasets/X_train_B.csv", "rb"), delimiter=",")
X_train_C = np.loadtxt(open("datasets/X_train_C.csv", "rb"), delimiter=",")
Y_test_A = np.loadtxt(open("datasets/Y_test_A.csv", "rb"), delimiter=",")
Y_test_B = np.loadtxt(open("datasets/Y_test_B.csv", "rb"), delimiter=",")
Y_test_C = np.loadtxt(open("datasets/Y_test_C.csv", "rb"), delimiter=",")
Y_train_A = np.loadtxt(open("datasets/Y_train_A.csv", "rb"), delimiter=",")
Y_train_B = np.loadtxt(open("datasets/Y_train_B.csv", "rb"), delimiter=",")
Y_train_C = np.loadtxt(open("datasets/Y_train_C.csv", "rb"), delimiter=",")

datasets = [[X_test_A,X_train_A,Y_test_A,Y_train_A],[X_test_B,X_train_B,Y_test_B,Y_train_B],[X_test_B,X_train_B,Y_test_B,Y_train_B]]

for X_test, X_train, y_test, y_train in datasets:
    # linear (unregularized) regression
    unreg_model = linear_model.LinearRegression()
    unreg_model.fit(X_train, y_train)
    pred_lin = unreg_model.predict(X_test)
    mse_lin = mean_squared_error(y_test, pred_lin)

    # ridge
    ridge_alphas = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
    max_score = 0
    best_alpha = 0
    for alpha in ridge_alphas:
        ridge_model = linear_model.Ridge(alpha = alpha)
        k_folds = KFold(n_splits=10)
        scores = cross_val_score(ridge_model, X_train, y_train, cv=k_folds)
        mean_score = np.mean(scores)
        max_score = max(mean_score, max_score)
        if max_score==mean_score:
            best_alpha = alpha
    best_ridge_model = linear_model.Ridge(alpha = best_alpha)
    best_ridge_model.fit(X_train, y_train)
    pred_ridge = best_ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, pred_ridge)

    # lasso
    lasso_alphas = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
    max_score = 0
    best_alpha = 0
    for alpha in lasso_alphas:
        lasso_model = linear_model.Lasso(alpha = alpha)
        k_folds = KFold(n_splits=10)
        scores = cross_val_score(lasso_model, X_train, y_train, cv=k_folds)
        mean_score = np.mean(scores)
        max_score = max(mean_score, max_score)
        if max_score==mean_score:
            best_alpha = alpha
    best_lasso_model = linear_model.Lasso(alpha = best_alpha)
    best_lasso_model.fit(X_train, y_train)
    pred_lasso = best_lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, pred_lasso)

    # avg mean squared error
    print("Average mean squared error of each model:")
    print("Unregularized linear model: ", mse_lin)
    print("Ridge regression model: ", mse_ridge)
    print("Lasso regression model: ", mse_lasso)

    # histogram
    plt.figure()
    plt.hist(unreg_model.coef_,alpha=0.5, label="Unregularized", bins=20, color="blue")
    plt.hist(best_ridge_model.coef_,alpha=0.5, label="Ridge", bins=20, color="red")
    plt.hist(best_lasso_model.coef_,alpha=0.5, label="Lasso", bins=20, color="black")
    plt.legend()
    plt.title("Parameter Histograms")
    plt.show()
