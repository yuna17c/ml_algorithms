import numpy as np
import matplotlib.pyplot as plt


# return the k smallest term in arr
def k_smallest(arr, k):
    pivot = arr[0]
    l = [ x for x in arr if x<pivot ]
    r = [ x for x in arr if x>pivot ]
    m = [ x for x in arr if x==pivot]
    if k<=len(l):
        return k_smallest(l, k)
    elif k>len(l)+len(m):
        return k_smallest(r, k-len(l)-len(m))
    else:
        return m[0]


# return the label using k-nearest neighbor regression
def knn_regression(X_arr, y_arr, val, k):
    n = len(X_arr)
    dist_arr = np.empty(n)
    for i in range(0, n):
        dist = np.linalg.norm(X_arr[i]-val)
        dist_arr[i] = dist
    kth = k_smallest(dist_arr, k)
    y_sum = 0
    cnt = k
    # average the y values of the k nearest neighbors
    for i in range(0, n):
        d = dist_arr[i]
        if d<kth:
            y_sum+=y_arr[i]
            cnt-=1
        elif d==kth and cnt!=0:
            y_sum+=y_arr[i]
            cnt-=1
    return y_sum/k


def plot_data(e, mse_knn, X_test, preds, graph_data=True):
    k_list = np.arange(1,10)
    # plot MSE: error of KNN for each k=1...9 and the error of least squares
    plt.figure()
    plt.scatter(k_list, mse_knn[1:], color="blue", label="Error of KNN")
    plt.axhline(y=e, label="Error of Least-Squares", color="red")
    plt.title("Error of KNN for each k and Least-Squares Solution")
    plt.legend()
    plt.show()

    # plot data
    if graph_data:
        plt.figure()
        plt.plot(X_test, preds[1], 'ro', label="1NN")
        plt.plot(X_test, preds[8], 'bo', label="9NN")
        plt.plot(X_test, preds[0], color='black', marker='o', label="Least-Squares")
        plt.title("1NN, 9NN, and Least-squares Solution")
        plt.legend()
        plt.show()


def run_model(X_train, y_train, X_test, y_test, graph_data=True):
    preds = []
    # Least squares linear regression
    if X_train.ndim==1:
        ybar = np.mean(y_train)
        xbar = np.mean(X_train)
        w = np.sum((y_train-ybar)*(X_train-xbar))/np.sum((X_train-xbar)**2)
        b = ybar - xbar*w
        y_pred = [ b+w * x for x in X_test]
    else:
        n, d = X_train.shape
        new_X_train = np.hstack((np.ones((n, 1)), X_train))
        response = np.linalg.solve(new_X_train.T @ new_X_train, new_X_train.T @ y_train)
        w = response[1:]
        b = response[0]
        y_pred = [ b+w @ x for x in X_test]
    preds.append(y_pred) 
    
    # Knn regression
    for k in range(1, 10):
        y_pred = [ knn_regression(X_train, y_train, x, k) for x in X_test ]
        preds.append(y_pred)

    # mse (error of knn)
    mse_knn = [ np.mean((prediction-y_test)**2) for prediction in preds ]

    # error of least squares
    e = np.mean(abs(y_test-preds[0])**2)

    # plot data and errors
    if graph_data==False:
        plot_data(e, mse_knn, X_test, preds, False)
    else:
        plot_data(e, mse_knn, X_test, preds)

