import numpy as np
import pandas as pd 
from functools import reduce

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

def gen_opt(data):
    fs = [v['fqc'] for v in data]
    fssum = sum(fs)
    opt_vec = []
    for i in data:
        i.update({'fqc': float(i['fqc'] / fssum)})
        opt_vec.append(i['fqc'])
    
    return opt_vec

def topsis(data):
    
    return

def ahp(data):
    
    return

if __name__ == "__main__":
    fpath = "./res.xlsx"
    keynum = 10 # number of selected keys
    # data[index]{['id']['key']['fqc']}
    # try norm weight
    data = load_data(fpath, keynum)
    opt = gen_opt(data)
    print(opt)
    print("Now check Consistency Ratio...")