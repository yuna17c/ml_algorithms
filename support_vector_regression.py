import sys
import numpy as np
import sys


def SVR(X_train, Y_train, C, eps, eta):
    max_pass = 1000
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0
    for t in range(0,max_pass):
        for i in range(0,n):
            y = Y_train[i]
            x = X_train[i]
            inner = y-(x@w+b)
            if inner>eps:
                w -= eta*(-C)*x
                b -= eta*(-C)
            elif inner<-eps:
                w -= eta*C*x
                b -= eta*C
            w = w/(1+eta)    # proximal step
    return w, b


def compute_loss(X, Y, w, b, C, eps):
    #Return: loss computed on the given set
    loss = w.T@w/2 + compute_error(X,Y,w,b,C,eps)
    return loss


def compute_error(X, Y, w, b, C, eps):
    #Return: error computed on the given set
    error = C*sum(max(abs(Y[i]-(X[i]@w+b))-eps,0) for i in range(len(Y)))
    return error


def find_eta(X_train, Y_train, C, eps):
    etas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
    min_e = -1
    for eta in etas:
        w, b = SVR(X_train, Y_train, C, eps, eta)
        test_error = compute_error(X_test, Y_test, w, b, C, eps)
        if min_e==-1 or min_e>test_error:
            min_e = test_error
            best_eta = eta
    return best_eta


def get_errors(X_train, Y_train, w, b, C, eps):
    train_error = compute_error(X_train, Y_train, w, b, C, eps)
    test_error = compute_error(X_test, Y_test, w, b, C, eps)
    train_loss = compute_loss(X_train, Y_train, w, b, C, eps)
    return train_error, train_loss, test_error


if __name__ == "__main__":
    args = sys.argv[1:]
    X_train = np.loadtxt(open("data/X_train_C.csv", "rb"), delimiter=",")
    Y_train = np.loadtxt(open("data/Y_train_C.csv", "rb"), delimiter=",")
    X_test = np.loadtxt(open("data/X_test_C.csv", "rb"), delimiter=",")
    Y_test = np.loadtxt(open("data/Y_test_C.csv", "rb"), delimiter=",")
    C = float(args[0])
    eps = float(args[1])
    best_eta = find_eta(X_train, Y_train, C, eps)
    w, b = SVR(X_train, Y_train, C, eps, best_eta)
    train_error, train_loss, test_error = get_errors(X_train, Y_train, w, b, C, eps)
    print("train error", train_error)
    print("test error", test_error)
    print("train loss", train_loss)
