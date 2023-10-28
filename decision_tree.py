import numpy as np
from math import *
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, feature_idx=0, threshold=0, left=None, right=None, val=None):
        self.feature_index = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val


class DecisionTree:
    def __init__(self, max_depth, loss_fn, random=False):
        self.max_depth = max_depth
        self.root = None
        self.loss_fn = loss_fn
        self.random = random        # random is true for random forest
        return
    
    # return loss depending on the loss function
    def calc_loss(self, idx, y):
        n = len(y[idx])
        loss = 0
        if n!=0:
            p = np.count_nonzero(y[idx] == 1)/n
            loss = len(idx[0])/len(y)
            if self.loss_fn=="misclassification":
                loss *= min(p, 1-p)
            elif self.loss_fn=="gini":
                loss *= (p*(1-p))
            elif self.loss_fn=="entropy":
                if p==1 or p==0:
                    loss = 0
                else:
                    loss *= ((-p)*log(p,2)-(1-p)*log(1-p,2))
        return loss
    
    # return the label based on majority value
    def calc_leaf(self, y):
        a = np.count_nonzero(y==1)
        b = np.count_nonzero(y==0)
        if a>b:
            return 1
        return 0
    
    def fit(self, X, y):
        self.root = self.build(X,y,0)

    def build(self, X, y, depth):
        n_sample, n_features = X.shape
        # no-split conditions
        if depth>=self.max_depth or len(np.unique(y))==1 or n_sample==0:
            value = self.calc_leaf(y)
            return Node(val=value)
        
        best_loss = -1
        for i in range(0,n_features):
            thresholds = np.unique(X[:,i])
            if self.random==True:
                # pick 4 random features to choose the threshold from (for random forest)
                random_features = random.sample(range(n_features), 4)
                thresholds = np.unique(X[:,random_features])
            ith_col = X[:, i]
            for threshold in thresholds:
                # left loss + right loss
                loss = self.calc_loss(np.where(ith_col<=threshold), y) + self.calc_loss(np.where(ith_col>threshold), y)
                if loss<best_loss or best_loss==-1:
                    # update if new minimum loss is found
                    best_loss = loss
                    best_threshold = threshold
                    feature_idx = i

        col = X[:, feature_idx]
        left_idx = np.where(col<=best_threshold)
        right_idx = np.where(col>best_threshold)
        # recurse on the left and right data
        left_tree = self.build(X[left_idx], y[left_idx], depth+1)
        right_tree = self.build(X[right_idx],y[right_idx], depth+1)
        new_node = Node(feature_idx = feature_idx, threshold=best_threshold, left=left_tree, right=right_tree)
        return new_node
    
    def predict(self, X):
        predictions = [ self.predict_data(self.root, x) for x in X ]
        return predictions
        
    def predict_data(self, curr, X):
        if curr.val==None:
            if X[curr.feature_index]<=curr.threshold:
                return self.predict_data(curr.left, X)
            return self.predict_data(curr.right, X)
        return curr.val 


#Load data
X_train = np.loadtxt('data/X_train_D.csv', delimiter=",")
y_train = np.loadtxt('data/y_train_D.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test_D.csv', delimiter=",")
y_test = np.loadtxt('data/y_test_D.csv', delimiter=",").astype(int)


def calc_accuracy(pred, y):
    accuracy = np.sum(y == pred)/len(y)
    return accuracy


def create_graphs(x,y,color):
    # get label to create legend
    label = "Testing Accuracy"
    if color=="red":
        label = "Training Accuracy"
    plt.scatter(x, y, color=color, label=label)
    # create lines that connect the points
    for i in range(len(x) - 1):
        plt.plot([x[i], x[i+1]], [y[i], y[i+1]], linestyle='--', color=color)


def graph(training, testing, title):
    depths = np.arange(15)
    create_graphs(depths, training, "red")
    create_graphs(depths, testing, "blue")
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()


def get_accuracy(classifier):
    classifier.fit(X_train, y_train)
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)
    # accuracy
    train = calc_accuracy(pred_train, y_train)
    test = calc_accuracy(pred_test, y_test)
    return train, test


# accuracy array using each loss function [misclassification, gini, entropy]
train_acc, test_acc = [[],[],[]], [[],[],[]]

# find the accuracies for each method and max depth
for i in range(0,15):
    misclass_classifier = DecisionTree(max_depth=i, loss_fn="misclassification")
    train, test = get_accuracy(misclass_classifier)
    train_acc[0].append(train)
    test_acc[0].append(test)
    gini_classifier = DecisionTree(max_depth=i, loss_fn="gini")
    train, test = get_accuracy(gini_classifier)
    train_acc[1].append(train)
    test_acc[1].append(test)
    entropy_classifier = DecisionTree(max_depth=i, loss_fn="entropy")
    train, test = get_accuracy(entropy_classifier)
    train_acc[2].append(train)
    test_acc[2].append(test)
    

graph(train_acc[0], test_acc[0], title="Misclassification")
graph(train_acc[1], test_acc[1], title="Gini Index")
graph(train_acc[2], test_acc[2], title="Entropy")


# return a list of indices of randomly picked 101 datasets picked with replacement
def get_ensemble():
    n,d=X_train.shape
    s_idx = []
    for i in range(101):
        # idx holds the indices of the random data that's picked with replacement
        idx = []
        for i in range(n):
            num = random.randint(0, n - 1)
            idx.append(num)
        s_idx.append(idx)
    return s_idx


def run_ensemble_models(classifier):
    accuracy_lst = []
    for i in range(11):
        pred_lst = []
        # recurse through 101 randomly picked dataset and get the predicted labels
        s_idx = get_ensemble()
        for idx in s_idx:
            X = X_train[idx]
            classifier.fit(X, y_train[idx])
            pred_test = classifier.predict(X_test)
            pred_lst.append(pred_test)
        
        n = len(pred_test)
        final_pred = []
        # find the most common label
        for i in range(n):
            y = [ row[i] for row in pred_lst]
            a = sum(1 for yi in y if yi == 1)
            b = sum(1 for yi in y if yi == 0)
            if a>b:
                final_pred.append(1)
            else:
                final_pred.append(0)
        
        # accuracy
        test = calc_accuracy(final_pred, y_test)
        accuracy_lst.append(test)

    return np.mean(accuracy_lst),np.median(accuracy_lst),np.max(accuracy_lst),np.min(accuracy_lst)


# bagging
print("Bagging implementation")
bagging_classifier = DecisionTree(max_depth=3, loss_fn="entropy")
mean, median, maximum, minimum = run_ensemble_models(bagging_classifier)
print(mean, median, maximum, minimum)


# random forest
print("Random forest")
rf_classifier = DecisionTree(max_depth=3, loss_fn="entropy", random=True)
mean, median, maximum, minimum = run_ensemble_models(rf_classifier)
print(mean, median, maximum, minimum)
