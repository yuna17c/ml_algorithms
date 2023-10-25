import numpy as np
import time
import matplotlib.pyplot as plt


class Ridge_Regression:
    def __init__(self, X, y, lmbd):
        n, d = X.shape
        I = np.identity(d)
        w = np.linalg.solve(X.T @ X + lmbd*I, X.T @ y)
        self.w = w
    
    def predict(self, X):
         y = X @ self.w.T
         return y


class Gradient_Descent:
    def __init__(self, X, y, lmbd, step_size, max_pass, tol):
        n, d = X.shape
        w = np.zeros(d)
        b = 0
        losses = []
        for t in range(0, max_pass):
            expr = X @ w + b*np.ones(n).T - y
            w_gradient = (X.T @ expr)/n + 2*lmbd*w
            b_gradient = (np.ones(n) @ expr)/n
            wt = w - step_size*w_gradient
            bt = b - step_size*b_gradient
            loss = np.mean((X @ wt + bt * np.ones(n).T - y) ** 2)
            losses.append(loss)
            if np.linalg.norm(wt-w)<tol:
                w = np.copy(wt)
                b = bt
                break
            w = np.copy(wt)
            b = bt
        self.w = w
        self.b = b
        self.losses = losses

    def predict(self, X):
         y = X @ self.w + self.b         
         return y
    

def calc_mse(pred, real):
    return np.mean(np.square(pred-real))


def print_val(model, time, train_error, test_error, train_loss=0):
    print("--"+model+"--")
    print("Running time: ", time)
    print(f"Training error: {train_error}")
    print(f"Test error: {test_error}")
    if model=="Gradient Descent":
        print(f"Training loss: {train_loss}")
    

def test_implementations(X_train, y_train, X_test, y_test, X_train_std, X_test_std):
    for lmbd in ([0, 10]):
        print(f"Lambda: {lmbd}")

        # ridge regression
        start_time = time.perf_counter()
        ridge_model = Ridge_Regression(X_train, y_train, lmbd)
        y_pred_train_ridge = ridge_model.predict(X_train)
        y_pred_ridge = ridge_model.predict(X_test)
        end_time = time.perf_counter()
        
        # errors
        train_error_ridge = calc_mse(y_pred_train_ridge, y_train)
        test_error_ridge = calc_mse(y_pred_ridge, y_test)
        print_val("Ridge Regression", end_time-start_time, train_error_ridge, test_error_ridge)

        # gradient descent
        start_time = time.perf_counter()
        step_size, max_pass, tol = 0.00001, 999, 0.000001
        gd_model = Gradient_Descent(X_train_std, y_train, lmbd, step_size, max_pass, tol)
        y_pred_train_gradient = gd_model.predict(X_train_std)
        y_pred_gradient = gd_model.predict(X_test_std)
        end_time = time.perf_counter()

        # errors
        train_error_gradient = calc_mse(y_pred_train_gradient,y_train)
        test_error_gradient = calc_mse(y_pred_gradient,y_test)
        print_val("Gradient Descent", end_time-start_time, train_error_gradient, test_error_gradient, gd_model.losses)

        # plot the training loss over iterations
        plt.plot(gd_model.losses)
        plt.show()
