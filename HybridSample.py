# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 09:24:13 2025

@author: 16235
"""
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import random

def undersample(X, y, k=5, C=5):
    Xy = np.column_stack((X, y)).tolist()
    knn = KNeighborsClassifier(n_neighbors=k+1)
    knn.fit(X, y)
    distances, indices = knn.kneighbors(X)
    kmeans = KMeans(n_clusters=C)
    kmeans.fit(X)
    y_km = kmeans.predict(X)
    Insts = []
    for indice in indices:
        idx = np.argwhere(y_km != y_km[indice[0]])
        Inst = np.reshape(idx, idx.shape[0]).tolist()
        Inst = [i for i in Inst if i in indice]
        Insts.append(Inst)
    # print(Insts)
    
    clustIns = {}
    cdict = {}
    cdict_index = {}
    
    for i in range(len(y_km)):
        ins = []
        ins.append(Insts[i])
        if y_km[i] not in clustIns:
            clustIns[y_km[i]] = ins
            cdict[y_km[i]] = [Xy[i]]
            cdict_index[y_km[i]] = [i]
        else:
            clustIns[y_km[i]] += ins
            cdict[y_km[i]].append(Xy[i])
            cdict_index[y_km[i]].append(i)
    # print(clustIns)
    # print(cdict)
    # print(cdict_index) 
        
    clustIns = dict(sorted(clustIns.items(), key=lambda item: item[0]))
    clusImp_r = [len(i)/C for i in Insts]
    # print(clusImp_r)
    clusImp = []
    for k,v in clustIns.items():
        N = []
        if k in y_km:
            for i in v:
                N.append(len(i)/C)
        else:
            N.append(0)
        clusImp.append(N)
    # print(clusImp)
    clusC_med = []
    for i in clusImp:
        clusC_med.append(np.average(i))
    # print(clusC_med)
    
    Sconj = []
    for i in range(len(clusImp_r)):
        clus_idx = y_km[i]
        if clusC_med[clus_idx] < clusImp_r[i]:
            Sconj.append(X[i])   
    # print(Sconj)
    
    #identify the dominant cluster
    clusDom = []
    noise = []

    for inst in range(len(Insts)):
        idx = [y_km[i] for i in Insts[inst]]
        # print(inst)
        # print(idx)
        if len(set(idx)) == 0:
            maxE = "nan"
        elif len(set(idx)) == 1:
            maxE = idx[0]
        else:
            result = Counter(idx)
            # print(result)
            if list(result.values())[1] == list(result.values())[0]:
                noise.append(X[inst])
                maxE = "nan"
            else:
                maxE = result.most_common(1)[0][0]
        clusDom.append(maxE)
    # print(clusDom)

    # print(len(Sconj), len(noise))
    noise = [x for x in noise if x not in Sconj]
    # print(len(Sconj), len(noise))
    # print(noise)
    
    indexs = {}
    for ne in noise:
        if len(noise) != 0:
            for key, value in cdict.items():
                vx = np.array(value)[:, :-1].tolist()      
                vy = np.array(value)[:, -1].tolist()
                indices = sorted([i for i, x in enumerate(vx) if x == ne], reverse=True)
                if len(indices) != 0:
                    indexs[key] = indexs.get(key, []) + indices
    # print(indexs)
    # print(cdict_index) 
    for k,v in indexs.items():
        for inx in set(v):
            cdict_index[k][inx] = str(cdict_index[k][inx])
    # print(cdict_index)                    
                        
    for ne in noise:
        if len(noise) != 0:
            for key, value in cdict.items():
                vx = np.array(value)[:, :-1].tolist()
                print(vx)
                vy = np.array(value)[:, -1].tolist()
                indices = sorted([i for i, x in enumerate(vx) if x == ne], reverse=True)
                print(indices)
                if len(indices) != 0:
                    for j in indices:
                        value.pop(j)
                    if len(value) == 0:
                        cdict[key] = [["nan"] + vy]
    
    # print(cdict)
    return cdict, cdict_index
    # print(cdict)
    
    # print(len([i for i in y if i == 0]))
    # if len(noise) >= len([i for i in y if i == 0]):
    #     return X, y
    # else:
    #     return cdict, 0

def oversample(cdict, cdict_index, data, label):
    label = label.tolist()
    #cdict = {c1:[[x1, y1],[x2, y2]],...} i.e. {0：[], 1:[], 2:[],...}
    # print(cdict_index)
    # Y = [i for i in label]
    Y = []
    for v in cdict.values():
        for x in v:
            if x[:-1] != ['nan']:
                Y.append(x[-1])
    # print("Y", Y)
    
    G = abs(np.sum(np.array(Y)) - (len(Y)-np.sum(np.array(Y))))
    print(G)
    
    IR = {}
    C = {}
    sums = 0
    for k,v in cdict.items():
        l = 0
        y = [i[-1] for i in v]
        y0 = [i for i in y if i == 1]
        y1 = [i for i in y if i == 0]
        print(len(y1), len(y0), len(y1) - len(y0))
        delta = abs(len(y1) - len(y0))
        if len(v) > 1:
            if len(y0) != 0 and delta != 0:
                sums += delta
                l = 1 / delta
            IR[k] = l
            C[k] = delta
        elif len(v) == 1:
            if len(y1) == 0:
                sums += delta
                l = 1 / delta
                IR[k] = 1
                C[k] = 1
            # elif len(y1) == 0:
            #     IR[k] = 1
            #     C[k] = 1
            
    
    # print("sum", sum)
    # print("IR", IR)

    std_IR = {}  
    for k,v in IR.items():
        if v > 0:
            std_IR[k] = 1 / (v * v * sums)
        else:
            std_IR[k] = 0.0
    # print("std_IR", std_IR)
    
    gc = {}
    for k,v in std_IR.items():
        c = C[k]
        gc[k] = int(1/c * v * G)
    # print("gc", gc)
    
    
    for k,v in gc.items():
        for _ in range(v):
            if v > 1:
                beta = random.random()
                index = cdict_index[k]
                s = []
                s.append(data[index[0]])
                s.append(data[index[1]]) 
                x = (np.array(s[0]) +  (np.array(s[1]) - np.array(s[0])) * beta).tolist()
                y = 1
                data.append(x)
                label.append(y)
            elif v == 1:
                la = set([i[-1] for i in cdict[k]])
                if len(la) > 1:
                    continue
                else:
                    beta = random.random()
                    index = cdict_index[k]
                    s = []
                    s.append(data[index[0]])
                    x = (np.array(s[0]) * (1-beta)).tolist()
                    y = 1
                    data.append(x)
                    label.append(y)
    indices = []
    for k in cdict.keys():
        v = cdict[k][0][0]
        if v == 'nan':
            f = [eval(i) for i in cdict_index[k]]
            indices += f
            
    for i in sorted(indices, reverse=True):
        del data[i]
        del label[i]
    print(len(data), len(label))
    return data, np.array(label)


# def oversample(cdict):
#     #cdict = {c1:[[x1, y1],[x2, y2]],...} i.e. {0：[], 1:[], 2:[],...}
#     X = []
#     Y = []
#     for v in cdict.values():
#         for x in v:
#             if x[:-1] != ['nan']:
#                 Y.append(x[-1])
#     # print("Y", Y)
    
#     G = abs(np.sum(np.array(Y)) - (len(Y)-np.sum(np.array(Y))))
#     print(G)
    
#     IR = {}
#     C = {}
#     # print(cdict)
#     sum = 0
#     for k,v in cdict.items():
#         l = 0
#         y = [i[-1] for i in v]
#         y0 = [i for i in y if i == 1]
#         y1 = [i for i in y if i == 0]
#         print(len(y1), len(y0), len(y1) - len(y0))
#         delta = abs(len(y1) - len(y0))
#         if len(v) > 1:
#             # y = [i[-1] for i in v]
#             # y0 = [i for i in y if i == 1]
#             # y1 = [i for i in y if i == 0]
#             if len(y0) != 0 and delta != 0:
#                 sum += delta
#                 l = 1 / delta
#             IR[k] = l
#             C[k] = delta
#         elif len(v) == 1:
#             # y = [i[-1] for i in v]
#             # y0 = [i for i in y if i == 1]
#             # y1 = [i for i in y if i == 0]
#             if len(y1) == 0:
#                 sum += delta
#                 l = 1 / delta
#                 IR[k] = 1
#                 C[k] = 1
#             # elif len(y1) == 0:
#             #     IR[k] = 1
#             #     C[k] = 1
            
    
#     print("sum", sum)
#     print("IR", IR)

#     std_IR = {}  
#     for k,v in IR.items():
#         if v > 0:
#             std_IR[k] = 1 / (v * v * sum)
#         else:
#             std_IR[k] = 0.0
#     print("std_IR", std_IR)
    
#     gc = {}
#     for k,v in std_IR.items():
#         c = C[k]
#         gc[k] = int(1/c * v * G)
#     print("gc", gc)
    
    
#     for k,v in gc.items():
#         for _ in range(v):
#             if v > 1:
#                 print("v=", v)
#                 beta = random.random()
#                 s = cdict[k]
#                 x = (np.array(s[0])[:-1] +  (np.array(s[1])[:-1] - np.array(s[0])[:-1]) * beta).tolist()
#                 y = [1.0]
#                 xy = x + y
#                 cdict[k].append(xy)
#             elif v == 1:
#                 print("v=", v)
#                 beta = random.random()
#                 s = cdict[k]
#                 x = (np.array(s[0])[:-1] * (1-beta)).tolist()
#                 y = [1.0]
#                 xy = x + y
#                 cdict[k].append(xy)
#     # print(cdict)
      # Y = []
#     for v in cdict.values():
#         for x in v:
#             # print(x[:-1])
#             if x[:-1] != ['nan']: 
#                 X.append(x[:-1])
#                 Y.append(x[-1])

    
    # return X, Y

def hybridsample(X, y):
    cdict, cdict_index = undersample(X, y)
    if isinstance(cdict, dict):
        X, y = oversample(cdict, cdict_index, X, y)
        return X, y
    else:
        return X, y
    

def fit(X_train, y_train, X_test, y_test, module="SL"):
    '''
    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    module : TYPE, optional
        DESCRIPTION. The default is "SL".

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    '''
    if module == "SL":
        acc_final = 0.0
        X_final = []
        y_final = []
        model = svm.SVC()
        model.fit(X_train, y_train)
        print(X_train, y_train)
        y_pred = model.predict(X_test)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
        print(tn, fp, fn, tp)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        acc = (sen * spe) ** 0.5
        print(len(X_train), len(y_train))
        print(acc)
        while abs(acc - acc_final) >= 0.00001:
            X_train_b, y_train_b = hybridsample(X_train, y_train)
            model.fit(X_train_b, y_train_b)
            print(X_train_b)
            y_pred = model.predict(X_test)
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            print(tn, fp, fn, tp)
            sen = tp / (tp + fn)
            spe = tn / (tn + fp)
            acc_final = (sen * spe) ** 0.5
            if acc <= acc_final:
                acc = acc_final
                X_final, y_final = X_train_b, y_train_b
            print(len(X_final), len(y_final))
            print(acc_final)
        return X_final, y_final
    elif module == "UL":
        train, y_train = hybridsample(X_train, y_train)
        return X_train, y_train
    else:
        print("Noncompliant input!!!")

def main():
    import sys
    sys.path.append(r'E:\赵思远资料\赵思远资料\文献\PAMPred-main\CircPePred-mian') 
    import data_processing
    from sklearn.model_selection import train_test_split
    path = 'E:/赵思远资料/赵思远资料/文献/PAMPred-main/CircPePred-mian'
    sORFs_posi_file = path + '/dataset3/dataset/sORFs-training.txt'
    sORFs_nega_file = path + '/dataset3/dataset/non-sORFs-training.txt'
    posi_aas_file = path + '/dataset3/dataset/peptides-training.txt'
    nega_aas_file = path + '/dataset3/dataset/non-peptides-training.txt'
    posi_sorfs_samples, nega_sorfs_samples, sorfs_f_index, sorfs_f_name = data_processing.get_sorfs_dataset(sORFs_posi_file, sORFs_nega_file)
    posi_aas_file, nega_aas_file = data_processing.get_aas(sORFs_posi_file, sORFs_nega_file, posi_aas_file, nega_aas_file)
    posi_aas_samples, nega_aas_samples, aas_f_index, aas_f_name = data_processing.get_aas_dataset(posi_aas_file, nega_aas_file)

    posi_sorfs_samples, nega_sorfs_samples, posi_aas_samples, nega_aas_samples = data_processing.conn_shuf_split_dataset(posi_sorfs_samples, nega_sorfs_samples, posi_aas_samples, nega_aas_samples)
    
    sorfs_samples = np.vstack((posi_sorfs_samples,nega_sorfs_samples))
    # aas_samples = np.vstack((posi_aas_samples, nega_aas_samples))
    
    sorfs_samples_X = sorfs_samples[:, :-1]
    sorfs_samples_y = sorfs_samples[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(
    sorfs_samples_X, sorfs_samples_y, 
    test_size=0.3, 
    random_state=42)
    
    return X_train, X_test, y_train, y_test

# X_train = [[0], [0.4], [1.7], [1.7], [1.9], [2], [3], [4], [5], [1.8], [2.5], [3.5], [4.5]]
# y_train = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# X_test, y_test = [[0], [0.4], [1.4], [1.85]], [1, 1, 0, 0]
# # # X_train, X_test, y_train, y_test = main()
# # # print(X_train.shape)
# X_train, y_train = fit(X_train, y_train, X_test, y_test, module="UL")
# print(X_train, y_train)