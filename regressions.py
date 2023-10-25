import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def calc_mse(pred, real):
    return np.mean(np.square(pred-real))


def split_dataset(x, y, i):
    n = x.shape[0]//10
    test_x = x[i*n:(i+1)*n]
    train_x = np.vstack((x[:i*n],x[(i+1)*n:]))
    test_y = y[i*n:(i+1)*n]
    train_y = np.hstack((y[:i*n],y[(i+1)*n:]))
    return train_x, test_x, train_y, test_y


# plot the histogram
def plot_data(unreg_coef,ridge_coef,lasso_coef):
    num_bins = 50
    plt.figure()
    plt.hist(unreg_coef,alpha=0.5, label="Unregularized", bins=num_bins, color="blue")
    plt.hist(ridge_coef,alpha=0.5, label="Ridge", bins=num_bins, color="red")
    plt.hist(lasso_coef,alpha=0.5, label="Lasso", bins=num_bins, color="black")
    plt.legend()
    plt.title("Parameter Histograms")
    plt.show()
    

def run_model(X_test, X_train, y_test, y_train):
    # linear (unregularized) regression
    unreg_model = linear_model.LinearRegression()
    unreg_model.fit(X_train, y_train)
    pred_lin = unreg_model.predict(X_test)
    mse_lin = calc_mse(y_test, pred_lin)

    # ridge
    best_alpha = 0
    max_score = -1
    # k-fold cross validation to find the best hyperparameter 
    for alpha in range(0,1000):
        if alpha==0:
            continue
        alpha = alpha/10.0
        scores = []
        for i in range(0,10):
            train_x, test_x, train_y, test_y = split_dataset(X_train, y_train, i)
            ridge_model = linear_model.Ridge(alpha = alpha)
            ridge_model.fit(train_x, train_y)
            y_pred = ridge_model.predict(test_x)
            scores.append(calc_mse(y_pred, test_y))
        mean_score = np.mean(scores)
        if (max_score==-1 or max_score>mean_score):
            max_score = mean_score 
            best_alpha = alpha
    ridge_model = linear_model.Ridge(alpha = best_alpha)
    ridge_model.fit(X_train, y_train)
    pred_ridge = ridge_model.predict(X_test)
    mse_ridge = calc_mse(y_test, pred_ridge)
    print("ridge hyperparameter: ", best_alpha)

    # lasso
    best_alpha = 0
    max_score = -1
    # k-fold cross validation to find the best hyperparameter
    for alpha in range(0,500):
        if alpha==0:
            continue
        alpha = alpha/100.0
        scores = []
        for i in range(0,10):
            train_x, test_x, train_y, test_y = split_dataset(X_train, y_train, i)
            lasso_model = linear_model.Lasso(alpha = alpha, tol=0.001)
            lasso_model.fit(train_x, train_y)
            y_pred = lasso_model.predict(test_x)
            scores.append(calc_mse(y_pred, test_y))
        mean_score = np.mean(scores)
        if (max_score==-1 or max_score>mean_score):
            max_score = mean_score 
            best_alpha = alpha
    lasso_model = linear_model.Lasso(alpha = best_alpha, tol=0.001)
    lasso_model.fit(X_train, y_train)
    pred_lasso = lasso_model.predict(X_test)
    mse_lasso = calc_mse(y_test, pred_lasso)
    print("lasso hyperparameter: ", best_alpha)

    # avg mean squared error
    print("Average mean squared error of each model:")
    print("Unregularized linear model: ", mse_lin)
    print("Ridge regression model: ", mse_ridge)
    print("Lasso regression model: ", mse_lasso)

    plot_data(unreg_model.coef_,ridge_model.coef_,lasso_model.coef_)

