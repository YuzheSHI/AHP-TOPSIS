import numpy as np
import pandas as pd 
from functools import reduce
import math

def load_data(path, keynum):
    res = pd.read_excel(
        path, 
        sheet_name = 0, 
        usecols = 'A : C'
    )
    res = res.to_dict(orient = 'index')
    get_keys = list(filter(lambda x: x < keynum, res.keys()))
    res = [dict(v) for k, v in res.items() if k in get_keys]
    return res

def load_res(path):
    
    keys = [
        'believable', 
        'delightable', 
        'playable', 
        'cool', 
        'stability', 
        'maturable', 
        'lovable', 
        'nobelity', 
        'convenience', 
        'brilliant'
        ]

    res1 = pd.read_excel(
        path, 
        sheet_name = 0, 
        usecols = 'J : T'
    )
    res2 = pd.read_excel(
        path, 
        sheet_name = 0, 
        usecols = 'U : AE'
    )
    res3 = pd.read_excel(
        path, 
        sheet_name = 0, 
        usecols = 'AF : AP'
    )
    res1 = res1.to_dict(orient = 'index')
    res2 = res2.to_dict(orient = 'index')
    res3 = res3.to_dict(orient = 'index')
    r = [res1, res2, res3]
    X1 = [0.0] * 10
    X2 = [0.0] * 10
    X3 = [0.0] * 10
    for k in keys:
        d = keys.index(k)
        for i in range(0, 30):
            X1[d] += res1[i][k]
            k2 = k + '.1'
            X2[d] += res2[i][k2]
            k3 = k + '.2'
            X3[d] += res3[i][k3]
    
    sum1 = sum(X1)
    X1 = list(map(lambda x: x / sum1, X1))
    sum2 = sum(X2)
    X2 = list(map(lambda x: x / sum2, X2))
    sum3 = sum(X3)
    X3 = list(map(lambda x: x / sum3, X3))

    return [X1, X2, X3]

def gen_opt(data):
    fs = [v['fqc'] for v in data]
    fssum = sum(fs)
    opt_vec = []
    for i in data:
        i.update({'fqc': float(i['fqc'] / fssum)})
        opt_vec.append(i['fqc'])
    
    return opt_vec

def topsis(W, res):
    # Normalize every column
    Z = np.matrix(res)
    z = np.linalg.norm(Z, axis = 0)
    # z_{ij} = z_{ij} / (\sum_{i=1}^n z_{ij}^2)
    Z = np.true_divide(Z, z)
    # Optimal Solution
    Z_pos = np.max(Z, axis = 0)
    # Worst Solution
    Z_neg = np.min(Z, axis = 0)
    # Now calculate distances to optimal and worst solution
    # dist_posi = \sqrt{\sum_{j=1}^m w_j(Z_posj - z_{ij})^2}
    # dist_negi = \sqrt{\sum_{j=1}^m w_j(Z_negj - z_{ij})^2}
    diff_pos = Z_pos - Z
    diff_neg = Z_neg - Z
    diff_pos = np.square(diff_pos)
    diff_neg = np.square(diff_neg)
    W = np.matrix(W)
    W = np.vstack([W, W, W])
    weighted_pos = np.multiply(W, diff_pos)
    weighted_neg = np.multiply(W, diff_neg)
    dist_pos = np.sqrt(np.sum(weighted_pos, axis = 1))
    dist_neg = np.sqrt(np.sum(weighted_neg, axis = 1))
    # calculate C
    C = np.true_divide(dist_pos, dist_pos + dist_neg)
    return C


if __name__ == "__main__":
    fpath = "./res.xlsx"
    keynum = 10 # number of selected keys
    # data[index]{['id']['key']['fqc']}
    # try norm weight
    data = load_data(fpath, keynum)
    rpath = "./ques.xlsx"
    res = load_res(rpath)
    weight = gen_opt(data)
    print("Now operate TOPSIS...")
    C = topsis(weight, res)
    C = list(map(lambda x: float(x), C))
    order = np.argsort(np.array(C))[::-1]
    for i in range(0, len(C)):
        print("The No.", order[i] + 1, "solution is the No.", i + 1, "optimal one,")
        print("with C =", C[order[i]])