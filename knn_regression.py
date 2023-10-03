import numpy as np


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

def knn_regression(X_arr, y_arr, val, k):
    n = len(X_arr)
    dist_arr = np.empty(n)
    for i in range(0, n):
        dist = np.linalg.norm(X_arr[i]-val)
        dist_arr[i] = dist
    kth = k_smallest(dist_arr, k)
    y_sum = 0
    cnt = k
    for i in range(0, n):
        d = dist_arr[i]
        if d<kth:
            y_sum+=y_arr[i]
            cnt-=1
        elif d==kth and cnt!=0:
            y_sum+=y_arr[i]
            cnt-=1

    return y_sum/k

