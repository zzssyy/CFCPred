# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 20:12:21 2025

@author: 16235
"""

import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm

from sklearn.model_selection import StratifiedKFold
from HybridSample import undersample, oversample
from FeatureDescriptor import ReadFileFromFasta, FeatureGenerator
from FeatureSelection import EFISS_ES_FeatureSelection
from MLModel import MetricsCalculate, TrainBaselineMLModelOnly, ModelFiltered

def Filter_Ensemble_Classifier(feature_name, train_x, train_y):
    model_weight = {}
    model_all = {}
    for key in model_obj_arr:
        clf_arr = TrainBaselineMLModelOnly(feature_name, train_x[:, index_arr[key]], train_y, key, 5, test=False)
        model_all[key] = clf_arr
        model_weight[key] = clf_arr[3]

    model_weight = ModelFiltered(model_all, model_weight)
    return model_weight

def Ensemble_Label_Pro_Obtained(valids, valid_Y, model_weight, valid_y_class_arr, valid_y_pro_1_arr, valid_y_pro_0_arr, vote = 'hard', count = None, threshold = None):
    model_obj_arr = [i for i in model_weight if i > 0]
    loc = np.where((np.array(model_weight) > 0))
    if vote == 'hard':
        predict_valid_y_class = []
        predict_valid_y_pro_1 = []
        for index in range(len(valids)):
    
            valid_y_class_1 = 0
            valid_y_pro_1 = 0
            valid_y_pro_0 = 0
            
            temp_count = np.sum(model_weight)
            valid_y_class_1 += np.sum(np.array(valid_y_class_arr)[:,index][loc])
            w = np.array([i/temp_count for i in model_weight])
            valid_y_pro_1 += np.sum(w * np.array(valid_y_pro_1_arr)[:, index])
            valid_y_pro_0 += np.sum(w * np.array(valid_y_pro_0_arr)[:, index])
    
            if len(model_obj_arr) % 2 == 0:
                if valid_y_class_1 > (len(model_obj_arr)) // 2:
                    predict_valid_y_class.append(1)
    
                if valid_y_class_1 == (len(model_obj_arr)) // 2:
                    if valid_y_pro_1 >= valid_y_pro_0:
                        predict_valid_y_class.append(1)
                    if valid_y_pro_1 < valid_y_pro_0:
                        predict_valid_y_class.append(0)
    
                if valid_y_class_1 < (len(model_obj_arr)) // 2:
                    predict_valid_y_class.append(0)
    
            if len(model_obj_arr) % 2 != 0:
                if valid_y_class_1 >= (len(model_obj_arr) + 1) // 2:
                    predict_valid_y_class.append(1)
                else:
                    predict_valid_y_class.append(0)
    
            predict_valid_y_pro_1.append(valid_y_pro_1)
    
        metrics_value, confusion = MetricsCalculate(valid_Y, np.array(predict_valid_y_class), np.array(predict_valid_y_pro_1))
    
        return metrics_value, confusion
    
    elif vote == 'soft':
        predict_valid_y_class = []
        predict_valid_y_pro_1 = []
        for index in range(len(valids)):

            valid_y_pro_1 = 0
            valid_y_pro_0 = 0
            
            temp_count = np.sum(model_weight)
            w = np.array([i/temp_count for i in model_weight])
            valid_y_pro_1 += np.sum(w * np.array(valid_y_pro_1_arr)[:, index])
            valid_y_pro_0 += np.sum(w * np.array(valid_y_pro_0_arr)[:, index])

            if valid_y_pro_1 >= valid_y_pro_0:
                predict_valid_y_class.append(1)

            if valid_y_pro_1 < valid_y_pro_0:
                predict_valid_y_class.append(0)

            predict_valid_y_pro_1.append(valid_y_pro_1)

        metrics_value, confusion = MetricsCalculate(valid_Y, np.array(predict_valid_y_class), np.array(predict_valid_y_pro_1))
        
        return metrics_value, confusion
    
    elif vote == 'devs':
        predict_valid_y_class_soft = []
        predict_valid_y_class_hard = []
        predict_valid_y_pro_1 = []
        for index in range(len(valids)):

            valid_y_class_1 = 0
            valid_y_pro_1 = 0
            valid_y_pro_0 = 0

            temp_count = np.sum(model_weight)
            valid_y_class_1 += np.sum(np.array(valid_y_class_arr)[:,index][loc])
            w = np.array([i/temp_count for i in model_weight])
            valid_y_pro_1 += np.sum(w * np.array(valid_y_pro_1_arr)[:, index])
            valid_y_pro_0 += np.sum(w * np.array(valid_y_pro_0_arr)[:, index])

            # soft
            if valid_y_pro_1 >= valid_y_pro_0:
                predict_valid_y_class_soft.append(1)

            if valid_y_pro_1 < valid_y_pro_0:
                predict_valid_y_class_soft.append(0)

            # hard
            if len(model_obj_arr) % 2 == 0:
                if valid_y_class_1 > (len(model_obj_arr)) / 2:
                    predict_valid_y_class_hard.append(1)

                if valid_y_class_1 == (len(model_obj_arr)) / 2:
                    if valid_y_pro_1 >= valid_y_pro_0:
                        predict_valid_y_class_hard.append(1)
                    if valid_y_pro_1 < valid_y_pro_0:
                        predict_valid_y_class_hard.append(0)

                if valid_y_class_1 < (len(model_obj_arr)) / 2:
                    predict_valid_y_class_hard.append(0)

            if len(model_obj_arr) % 2 != 0:
                if valid_y_class_1 >= (len(model_obj_arr) + 1) / 2:
                    predict_valid_y_class_hard.append(1)
                else:
                    predict_valid_y_class_hard.append(0)

            predict_valid_y_pro_1.append(valid_y_pro_1)

        metrics_value_hard, confusion_hard = MetricsCalculate(valid_Y, np.array(predict_valid_y_class_hard), np.array(predict_valid_y_pro_1))
        metrics_value_soft, confusion_soft = MetricsCalculate(valid_Y, np.array(predict_valid_y_class_soft), np.array(predict_valid_y_pro_1))

        if metrics_value_hard[3] >= metrics_value_soft[3]:
            predict_valid_y_class = predict_valid_y_class_hard
        else:
            predict_valid_y_class = predict_valid_y_class_soft

        for index in range(len(valids)):

            valid_y_class_1 = 0
            valid_y_pro_0 = 0
            valid_y_pro_1 = 0
            flag = 0
            
            DC_ds = []
            model_u = abs(np.array(valid_y_pro_1_arr)[:, index] - np.array(valid_y_pro_0_arr)[:, index])

            for i in range(len(model_u)):
                if i in loc[0]:
                    if model_u[i] <= threshold:
                        flag += 1
                        DC_ds.append(model_weight[i] + model_u[i])
            
            if len(DC_ds) == 0:
                continue
            else:
                DC_sum = np.sum(DC_ds)
                
                DC_dict = {}
                for i in range(len(DC_ds)):
                    DC_dict[i] = DC_ds[i] / DC_sum
    
                model_obj_arr_ranked = sorted(DC_dict.items(), key=lambda d: d[1], reverse=True)
    
                if flag >= (len(model_obj_arr) / 2):
    
                    temp_count = 0
                    for model in model_obj_arr_ranked[:count]:
                        temp_count = temp_count + model[1]
    
                    for model in model_obj_arr_ranked[:count]:
                        valid_y_pro_0 = valid_y_pro_0 + (model[1] / temp_count) * valid_y_pro_0_arr[model[0]][index]
                        valid_y_pro_1 = valid_y_pro_1 + (model[1] / temp_count) * valid_y_pro_1_arr[model[0]][index]
    
                    if valid_y_pro_1 >= valid_y_pro_0:
                        predict_valid_y_class[index] = 1
    
                    if valid_y_pro_1 < valid_y_pro_0:
                        predict_valid_y_class[index] = 0
    
                    predict_valid_y_pro_1[index] = valid_y_pro_1

        metrics_value, confusion = MetricsCalculate(valid_Y, np.array(predict_valid_y_class), np.array(predict_valid_y_pro_1))
        return metrics_value, confusion
    
def Ensemble_Classifier_CV(train_x, train_y, cv_fold, index_arr: dict, model_obj_arr):
    valid_y_class_arr_all = {}
    valid_y_pro_1_arr_all = {}
    valid_y_pro_0_arr_all = {}
    
    valid_y_class_arr_fold = {}
    valid_y_pro_1_arr_fold = {}
    valid_y_pro_0_arr_fold = {}
    
    ml_dict = {'NB': GaussianNB(),
               'RF': RandomForestClassifier(random_state=100),
               'GBDT': GradientBoostingClassifier(random_state=100),
               'SVM': svm.SVC(random_state=100, probability=True)}

    
    # model_weight = Filter_Ensemble_Classifier(train_x, train_y)
    # model_all = {}
    # for key in model_obj_arr:
    #     clf_arr = TrainBaselineMLModelOnly(train_x[:, index_arr[key]], train_y, key, 10, test=False)
    #     model_all[key] = clf_arr
    #     model_weight[key] = clf_arr[3]

    # arr_valid = []
    valids = []
    Y = []
    folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=100).split(train_x, train_y)

    for i, (train, valid) in enumerate(folds):

        valids += valid.tolist()
        train_X, train_Y = train_x[train], train_y[train]
        valid_X, valid_Y = train_x[valid], train_y[valid]
        Y += valid_Y.tolist()

        valid_y_class_arr = {}
        valid_y_pro_1_arr = {}
        valid_y_pro_0_arr = {}

        for key in model_obj_arr:
            model = ml_dict[key]
            model.fit(train_X[:, index_arr[key]], train_Y)
            valid_y_class_arr[key] = np.array(model.predict(valid_X[:, index_arr[key]]))
            valid_y_pro_1_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 1]
            valid_y_pro_0_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 0]

        valid_y_class_arr_fold[i] = valid_y_class_arr
        valid_y_pro_1_arr_fold[i] = valid_y_pro_1_arr
        valid_y_pro_0_arr_fold[i] = valid_y_pro_0_arr

    
        valid_y_class_arr_all = get_arr_all(model_obj_arr, valid_y_class_arr_fold)
        valid_y_pro_1_arr_all = get_arr_all(model_obj_arr, valid_y_pro_1_arr_fold)
        valid_y_pro_0_arr_all = get_arr_all(model_obj_arr, valid_y_pro_0_arr_fold)

    #     if vote == 'hard' or "devs":

    #         valid_y_class_arr = {}
    #         valid_y_pro_1_arr = {}
    #         valid_y_pro_0_arr = {}

    #         for key in model_obj_arr:
    #             model = ml_dict[key]
    #             model.fit(train_X[:, index_arr[key]], train_Y)
    #             valid_y_class_arr[key] = np.array(model.predict(valid_X[:, index_arr[key]]))
    #             valid_y_pro_1_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 1]
    #             valid_y_pro_0_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 0]

    #         valid_y_class_arr_fold[i] = valid_y_class_arr
    #         valid_y_pro_1_arr_fold[i] = valid_y_pro_1_arr
    #         valid_y_pro_0_arr_fold[i] = valid_y_pro_0_arr

    #     if vote == 'soft': 

    #         valid_y_class_arr = {}
    #         valid_y_pro_1_arr = {}
    #         valid_y_pro_0_arr = {}

    #         for key in model_obj_arr:
    #             model = ml_dict[key]
    #             model.fit(train_X[:, index_arr[key]], train_Y)
    #             valid_y_pro_1_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 1]
    #             valid_y_pro_0_arr[key] = np.array(model.predict_proba(valid_X[:, index_arr[key]]))[:, 0]

    #         valid_y_class_arr_fold[i] = valid_y_class_arr
    #         valid_y_pro_1_arr_fold[i] = valid_y_pro_1_arr
    #         valid_y_pro_0_arr_fold[i] = valid_y_pro_0_arr
    
    # if vote == "hard" or vote == "devs":
    #     valid_y_class_arr_all = get_arr_all(model_obj_arr, valid_y_class_arr_fold)
    #     valid_y_pro_1_arr_all = get_arr_all(model_obj_arr, valid_y_pro_1_arr_fold)
    #     valid_y_pro_0_arr_all = get_arr_all(model_obj_arr, valid_y_pro_0_arr_fold)
    # elif vote == "soft":
    #     valid_y_pro_1_arr_all = get_arr_all(model_obj_arr, valid_y_pro_1_arr_fold)
    #     valid_y_pro_0_arr_all = get_arr_all(model_obj_arr, valid_y_pro_0_arr_fold)
    
    return valid_y_class_arr_all, valid_y_pro_1_arr_all, valid_y_pro_0_arr_all, valids

def get_arr_all(model_obj_arr, dicts):
    dicts_all = {}
    for key in dicts.keys():
        valid_y_class_arr = dicts[key]
        for model in model_obj_arr:
            if model not in dicts_all:
                dicts_all[model] = valid_y_class_arr[model].tolist()
            else:
                dicts_all[model] += valid_y_class_arr[model].tolist()
    return dicts_all

def Ensemble_Classifier_Result(valid_y_class_arr_all, valid_y_pro_1_arr_all, valid_y_pro_0_arr_all, model_weight_all, train_labels, valids, vote = 'hard', count = None, threshold = None):
    """
    valids = [index,index,index,...,index]
    valid_y_class_arr_all = {"aac":{"NB":[1,1,0,...], "RF":[1,1,0,...],...,"SVM":[1,1,0,...]}, "dpc":{},..., "ctdd":{}}
    valid_y_pro_1_arr_all = {"aac":{"NB":[1,1,0,...], "RF":[1,1,0,...],...,"SVM":[1,1,0,...]}, "dpc":{},..., "ctdd":{}}
    valid_y_pro_0_arr_all = {"aac":{"NB":[1,1,0,...], "RF":[1,1,0,...],...,"SVM":[1,1,0,...]}, "dpc":{},..., "ctdd":{}}
    model_weight_all = {"aac":{"NB": , "RF": ,...,"SVM": }, "dpc":{},..., "ctdd":{}}
    """
    valid_y = [train_labels[i] for i in valids]
    
    model_weight = []
    for value in model_weight_all.values():
        for v in value.values():
            model_weight.append(v)
    
    valid_y_class_arr = []
    for value in valid_y_class_arr_all.values():
        for v in value.values():
            valid_y_class_arr.append(v)
        
    valid_y_pro_1_arr = []
    for value in valid_y_pro_1_arr_all.values():
        for v in value.values():
            valid_y_pro_1_arr.append(v)
            
    valid_y_pro_0_arr = []
    for value in valid_y_pro_0_arr_all.values():
        for v in value.values():
            valid_y_pro_0_arr.append(v)
            
    valid_scores, confusion = Ensemble_Label_Pro_Obtained(valids, valid_y, model_weight, valid_y_class_arr, valid_y_pro_1_arr, valid_y_pro_0_arr, vote = vote, count = count, threshold = threshold)
    print("validation_dataset_scores: ", valid_scores)
    print("validation_dataset_confusion: ", confusion)

def Ensemble_Classifier_Test(train_x, train_y, test_x, test_y, index_arr: dict, model_obj_arr, vote, count, threshold):
    ml_dict = {'NB': GaussianNB(),
               'RF': RandomForestClassifier(random_state=100),
               'GBDT': GradientBoostingClassifier(random_state=100),
               'SVM': svm.SVC(random_state=100, probability=True)}

    model_weight = {}
    for key in model_obj_arr:
        clf_arr = TrainBaselineMLModelOnly(train_x[:, index_arr[key]], train_y, key, 5, test=False)
        model_weight[key] = clf_arr[3]

    if vote == 'hard':

        for test_dataset in test_x.keys():
            predict_test_y_class = []
            predict_test_y_pro_1 = []

            test_y_class_arr = {}
            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_class_arr[key] = np.array(model.predict(test_x[test_dataset][:, index_arr[key]]))
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_class_1 = 0
                test_y_pro_1 = 0
                test_y_pro_0 = 0

                temp_count = 0
                for key in model_obj_arr:
                    temp_count = temp_count + model_weight[key]

                for key in model_obj_arr:
                    test_y_class_1 = test_y_class_1 + test_y_class_arr[key][index]
                    test_y_pro_1 = test_y_pro_1 + (model_weight[key] / temp_count) * test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + (model_weight[key] / temp_count) * test_y_pro_0_arr[key][index]

                if len(model_obj_arr) % 2 == 0:
                    if test_y_class_1 > (len(model_obj_arr)) // 2:
                        predict_test_y_class.append(1)

                    if test_y_class_1 == (len(model_obj_arr)) // 2:
                        if test_y_pro_1 >= test_y_pro_0:
                            predict_test_y_class.append(1)
                        if test_y_pro_1 < test_y_pro_0:
                            predict_test_y_class.append(0)

                    if test_y_class_1 < (len(model_obj_arr)) // 2:
                        predict_test_y_class.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if test_y_class_1 >= (len(model_obj_arr) + 1) // 2:
                        predict_test_y_class.append(1)
                    else:
                        predict_test_y_class.append(0)

                predict_test_y_pro_1.append(test_y_pro_1)

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))
            print("test_dataset_scores: ", metrics_value, confusion)

    if vote == 'soft':
        for test_dataset in test_x.keys():
            # print(test_dataset)
            predict_test_y_class = []
            predict_test_y_pro_1 = []

            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_pro_1 = 0
                test_y_pro_0 = 0

                temp_count = 0
                for key in model_obj_arr:
                    temp_count = temp_count + model_weight[key]

                for key in model_obj_arr:
                    test_y_pro_1 = test_y_pro_1 + (model_weight[key] / temp_count) * test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + (model_weight[key] / temp_count) * test_y_pro_0_arr[key][index]

                if test_y_pro_1 >= test_y_pro_0:
                    predict_test_y_class.append(1)

                if test_y_pro_1 < test_y_pro_0:
                    predict_test_y_class.append(0)

                predict_test_y_pro_1.append(test_y_pro_1)

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))
            print("test_dataset_scores: ", metrics_value, confusion)

    if vote == 'devs':

        for test_dataset in test_x.keys():

            # print(test_dataset)

            predict_test_y_class_hard = []
            predict_test_y_pro_1 = []

            test_y_class_arr = {}
            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_class_arr[key] = np.array(model.predict(test_x[test_dataset][:, index_arr[key]]))
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_class_1 = 0
                test_y_pro_1 = 0
                test_y_pro_0 = 0

                temp_count = 0
                for key in model_obj_arr:
                    temp_count = temp_count + model_weight[key]

                for key in model_obj_arr:
                    test_y_class_1 = test_y_class_1 + test_y_class_arr[key][index]
                    test_y_pro_1 = test_y_pro_1 + (model_weight[key] / temp_count) * test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + (model_weight[key] / temp_count) * test_y_pro_0_arr[key][index]

                # hard
                if len(model_obj_arr) % 2 == 0:
                    if test_y_class_1 > (len(model_obj_arr)) / 2:
                        predict_test_y_class_hard.append(1)

                    if test_y_class_1 == (len(model_obj_arr)) / 2:
                        if test_y_pro_1 >= test_y_pro_0:
                            predict_test_y_class_hard.append(1)
                        if test_y_pro_1 < test_y_pro_0:
                            predict_test_y_class_hard.append(0)

                    if test_y_class_1 < (len(model_obj_arr)) / 2:
                        predict_test_y_class_hard.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if test_y_class_1 >= (len(model_obj_arr) + 1) / 2:
                        predict_test_y_class_hard.append(1)
                    else:
                        predict_test_y_class_hard.append(0)

                predict_test_y_pro_1.append(test_y_pro_1)

            predict_test_y_class = predict_test_y_class_hard

            for index in range(len(test_x[test_dataset])):

                test_y_pro_0 = 0
                test_y_pro_1 = 0

                flag = 0

                model_u = {}

                DC_dict = {}

                for key in model_obj_arr:
                    model_u[key] = abs(test_y_pro_1_arr[key][index] - test_y_pro_0_arr[key][index])

                for key in model_obj_arr:
                    if model_u[key] <= threshold:
                        flag += 1
                    DC_dict[key] = model_weight[key] + model_u[key]

                DC_sum = sum(DC_dict.values())

                for key in model_obj_arr:
                    DC_dict[key] = DC_dict[key] / DC_sum

                model_obj_arr_ranked = sorted(DC_dict.items(), key=lambda d: d[1], reverse=True)

                if flag >= (len(model_obj_arr) / 2):

                    temp_count = 0
                    for model in model_obj_arr_ranked[:count]:
                        temp_count = temp_count + model[1]

                    for model in model_obj_arr_ranked[:count]:
                        test_y_pro_0 = test_y_pro_0 + (model[1] / temp_count) * test_y_pro_0_arr[model[0]][index]
                        test_y_pro_1 = test_y_pro_1 + (model[1] / temp_count) * test_y_pro_1_arr[model[0]][index]

                    if test_y_pro_1 >= test_y_pro_0:
                        predict_test_y_class[index] = 1

                    if test_y_pro_1 < test_y_pro_0:
                        predict_test_y_class[index] = 0

                    predict_test_y_pro_1[index] = test_y_pro_1

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))

            print("test_dataset_scores: ", metrics_value, confusion)

def Ensemble_Classifier_Pred(train_x, train_y, test_x, index_arr: dict, model_obj_arr, vote, count, threshold):
    ml_dict = {'NB': GaussianNB(),
               'RF': RandomForestClassifier(random_state=100),
               'GBDT': GradientBoostingClassifier(random_state=100),
               'SVM': svm.SVC(random_state=100, probability=True)}

    model_weight = {}
    for key in model_obj_arr:
        clf_arr = TrainBaselineMLModelOnly(train_x[:, index_arr[key]], train_y, key, 5, test=False)
        model_weight[key] = clf_arr[3]

    if vote == 'hard':

        for test_dataset in test_x.keys():
            # print(test_dataset)
            predict_test_y_class = []
            predict_test_y_pro_1 = []

            test_y_class_arr = {}
            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_class_arr[key] = np.array(model.predict(test_x[test_dataset][:, index_arr[key]]))
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_class_1 = 0
                test_y_pro_1 = 0
                test_y_pro_0 = 0

                temp_count = 0
                for key in model_obj_arr:
                    temp_count = temp_count + model_weight[key]

                for key in model_obj_arr:
                    test_y_class_1 = test_y_class_1 + test_y_class_arr[key][index]
                    test_y_pro_1 = test_y_pro_1 + (model_weight[key] / temp_count) * test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + (model_weight[key] / temp_count) * test_y_pro_0_arr[key][index]

                if len(model_obj_arr) % 2 == 0:
                    if test_y_class_1 > (len(model_obj_arr)) // 2:
                        predict_test_y_class.append(1)

                    if test_y_class_1 == (len(model_obj_arr)) // 2:
                        if test_y_pro_1 >= test_y_pro_0:
                            predict_test_y_class.append(1)
                        if test_y_pro_1 < test_y_pro_0:
                            predict_test_y_class.append(0)

                    if test_y_class_1 < (len(model_obj_arr)) // 2:
                        predict_test_y_class.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if test_y_class_1 >= (len(model_obj_arr) + 1) // 2:
                        predict_test_y_class.append(1)
                    else:
                        predict_test_y_class.append(0)

                predict_test_y_pro_1.append(test_y_pro_1)

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))
            print("test_dataset_scores: ", metrics_value, confusion)

    if vote == 'soft':
        for test_dataset in test_x.keys():
            # print(test_dataset)
            predict_test_y_class = []
            predict_test_y_pro_1 = []

            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_pro_1 = 0
                test_y_pro_0 = 0

                temp_count = 0
                for key in model_obj_arr:
                    temp_count = temp_count + model_weight[key]

                for key in model_obj_arr:
                    test_y_pro_1 = test_y_pro_1 + (model_weight[key] / temp_count) * test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + (model_weight[key] / temp_count) * test_y_pro_0_arr[key][index]

                if test_y_pro_1 >= test_y_pro_0:
                    predict_test_y_class.append(1)

                if test_y_pro_1 < test_y_pro_0:
                    predict_test_y_class.append(0)

                predict_test_y_pro_1.append(test_y_pro_1)

            metrics_value, confusion = MetricsCalculate(test_y[test_dataset], np.array(predict_test_y_class), np.array(predict_test_y_pro_1))
            print("test_dataset_scores: ", metrics_value, confusion)

    if vote == 'devs':

        for test_dataset in test_x.keys():

            # print(test_dataset)

            predict_test_y_class_hard = []
            predict_test_y_pro_1 = []

            test_y_class_arr = {}
            test_y_pro_1_arr = {}
            test_y_pro_0_arr = {}

            for key in model_obj_arr:
                model = ml_dict[key]
                model.fit(train_x[:, index_arr[key]], train_y)
                test_y_class_arr[key] = np.array(model.predict(test_x[test_dataset][:, index_arr[key]]))
                test_y_pro_1_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 1]
                test_y_pro_0_arr[key] = np.array(model.predict_proba(test_x[test_dataset][:, index_arr[key]]))[:, 0]

            for index in range(len(test_x[test_dataset])):

                test_y_class_1 = 0
                test_y_pro_1 = 0
                test_y_pro_0 = 0

                temp_count = 0
                for key in model_obj_arr:
                    temp_count = temp_count + model_weight[key]

                for key in model_obj_arr:
                    test_y_class_1 = test_y_class_1 + test_y_class_arr[key][index]
                    test_y_pro_1 = test_y_pro_1 + (model_weight[key] / temp_count) * test_y_pro_1_arr[key][index]
                    test_y_pro_0 = test_y_pro_0 + (model_weight[key] / temp_count) * test_y_pro_0_arr[key][index]

                # hard
                if len(model_obj_arr) % 2 == 0:
                    if test_y_class_1 > (len(model_obj_arr)) / 2:
                        predict_test_y_class_hard.append(1)

                    if test_y_class_1 == (len(model_obj_arr)) / 2:
                        if test_y_pro_1 >= test_y_pro_0:
                            predict_test_y_class_hard.append(1)
                        if test_y_pro_1 < test_y_pro_0:
                            predict_test_y_class_hard.append(0)

                    if test_y_class_1 < (len(model_obj_arr)) / 2:
                        predict_test_y_class_hard.append(0)

                if len(model_obj_arr) % 2 != 0:
                    if test_y_class_1 >= (len(model_obj_arr) + 1) / 2:
                        predict_test_y_class_hard.append(1)
                    else:
                        predict_test_y_class_hard.append(0)

                predict_test_y_pro_1.append(test_y_pro_1)

            predict_test_y_class = predict_test_y_class_hard

            for index in range(len(test_x[test_dataset])):

                test_y_pro_0 = 0
                test_y_pro_1 = 0

                flag = 0

                model_u = {}

                DC_dict = {}

                for key in model_obj_arr:
                    model_u[key] = abs(test_y_pro_1_arr[key][index] - test_y_pro_0_arr[key][index])

                for key in model_obj_arr:
                    if model_u[key] <= threshold:
                        flag += 1
                    DC_dict[key] = model_weight[key] + model_u[key]

                DC_sum = sum(DC_dict.values())

                for key in model_obj_arr:
                    DC_dict[key] = DC_dict[key] / DC_sum

                model_obj_arr_ranked = sorted(DC_dict.items(), key=lambda d: d[1], reverse=True)

                if flag >= (len(model_obj_arr) / 2):

                    temp_count = 0
                    for model in model_obj_arr_ranked[:count]:
                        temp_count = temp_count + model[1]

                    for model in model_obj_arr_ranked[:count]:
                        test_y_pro_0 = test_y_pro_0 + (model[1] / temp_count) * test_y_pro_0_arr[model[0]][index]
                        test_y_pro_1 = test_y_pro_1 + (model[1] / temp_count) * test_y_pro_1_arr[model[0]][index]

                    if test_y_pro_1 >= test_y_pro_0:
                        predict_test_y_class[index] = 1

                    if test_y_pro_1 < test_y_pro_0:
                        predict_test_y_class[index] = 0

                    predict_test_y_pro_1[index] = test_y_pro_1

            pred_positive_sample_index = []
            for i in range(len(predict_test_y_class)):
                if predict_test_y_class[i] == 1:
                    pred_positive_sample_index.append(i)
            # print(len(pred_positive_sample_index))
            # print(pred_positive_sample_index)

if __name__ == '__main__':

    print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()


    # ath_train_seq = ReadFileFromFasta("dataset\\ath_training_dataset_800.txt")
    ath_train_seq = ReadFileFromFasta("E:/赵思远资料/赵思远资料/文献/PAMPred-main/CircPePred-mian/dataset3/dataset/all-sORFs-training.txt")

    ath_imbalanced_test_seq = ReadFileFromFasta("dataset\\ath_independent_testing_dataset1.txt")
    gma_imbalanced_test_seq = ReadFileFromFasta("dataset\\gma_independent_testing_dataset2.txt")
    vvi_imbalanced_test_seq = ReadFileFromFasta("dataset\\vvi_independent_testing_dataset3.txt")
    hybrid_species = ReadFileFromFasta("dataset\\verified_independent_testing_dataset4.txt")

    # 特征生成
    train_features_sORFs, train_features_aas, train_labels, train_features_name = FeatureGenerator(ath_train_seq)
    independent_test1_features, independent_test1_labels, independent_test1_features_name = FeatureGenerator(ath_imbalanced_test_seq, 'independent-test')
    independent_test2_features, independent_test2_labels, independent_test2_features_name = FeatureGenerator(gma_imbalanced_test_seq, 'independent-test')
    independent_test3_features, independent_test3_labels, independent_test3_features_name = FeatureGenerator(vvi_imbalanced_test_seq, 'independent-test')
    independent_test4_features, independent_test4_labels, independent_test4_features_name = FeatureGenerator(hybrid_species, 'independent-test')
    
    
    # undersample for sORFs
    train_features_fss = []
    train_features_len = {}
    for key in train_features_sORFs.keys():
        
        if key in ['ESMER3', 'ESMER4', 'ESMER5', 'kmer', 'ssm', 'ggap1', 'ggap2']:
            train_features_fs = train_features_sORFs[key]
            train_features_len[key] = len(train_features_fs[0])
            if len(train_features_fss) == 0:
                train_features_fss += train_features_fs.tolist()
            else:
                train_features_fss = np.hstack((train_features_fss, train_features_fs)).tolist()
    cdict, cdict_index = undersample(train_features_fss, train_labels)
    print(train_features_len)
    
    # oversample for aas
    train_features_fss = []
    train_features_len = {}
    for key in train_features_aas.keys():
        
        if key not in ['ESMER3', 'ESMER4', 'ESMER5', 'kmer', 'ssm', 'ggap1', 'ggap2']:
            print('key: ', key)
            train_features_fs = train_features_aas[key]
            train_features_len[key] = len(train_features_fs[0])
            if len(train_features_fss) == 0:
                train_features_fss += train_features_fs.tolist()
            else:
                train_features_fss = np.hstack((train_features_fss, train_features_fs)).tolist()
    
    org = len(train_features_fss)
    if isinstance(cdict, dict):
        train_features_fss, train_labels = oversample(cdict, cdict_index, train_features_fss, train_labels)
    
    for fea in train_features_fss[org:]:
        sums = 0
        for key in train_features_len.keys():
            train_feature_fs = fea[sums : sums + train_features_len[key]]
            train_features_aas[key] = np.vstack((train_features_aas[key], train_feature_fs))
            sums += train_features_len[key]

    # optimal_feature = {'ESMER3': [18, 27, 39, 34], 'ESMER4': [20, 115, 54, 60], 'ESMER5': [196, 170, 191, 133]}

    # feature_quantity = optimal_feature['ESMER5']

    index_arr = {}
    model_weight_all = {}
    valids = []
    valid_y_class_arr_all_f = {}
    valid_y_pro_1_arr_all_f = {}
    valid_y_pro_0_arr_all_f = {}

    for key in train_features_aas.keys():

        if key not in ['ESMER3', 'ESMER4', 'ESMER5', 'kmer', 'ssm', 'ggap1', 'ggap2']:
            print('key: ', key)
            filtered_feature = EFISS_ES_FeatureSelection(train_features_aas[key], train_labels)
            
            index_arr['NB'] = filtered_feature[:]
            index_arr['SVM'] = filtered_feature[:]
            index_arr['RF'] = filtered_feature[:]
            index_arr['GBDT'] = filtered_feature[:]

            # index_arr['NB'] = filtered_feature[: feature_quantity[0]]
            # index_arr['SVM'] = filtered_feature[: feature_quantity[1]]
            # index_arr['RF'] = filtered_feature[: feature_quantity[2]]
            # index_arr['GBDT'] = filtered_feature[: feature_quantity[3]]

            train_features_fs = train_features_aas[key]

            test_x = {'independent_test1': independent_test1_features[key],
                      'independent_test2': independent_test2_features[key],
                      'independent_test3': independent_test3_features[key],
                      'independent_test4': independent_test4_features[key]}
            test_y = {'independent_test1': independent_test1_labels,
                      'independent_test2': independent_test2_labels,
                      'independent_test3': independent_test3_labels,
                      'independent_test4': independent_test4_labels}

            model_obj_arr = ['SVM', 'RF', 'GBDT', 'NB']
            
            model_weight = Filter_Ensemble_Classifier(key, train_features_fs, train_labels)
            model_weight_all[key] = model_weight
            
            valid_y_class_arr_all, valid_y_pro_1_arr_all, valid_y_pro_0_arr_all, valids = Ensemble_Classifier_CV(train_features_fs, train_labels, 5, index_arr, model_obj_arr)
            valid_y_class_arr_all_f[key] = valid_y_class_arr_all
            valid_y_pro_1_arr_all_f[key] = valid_y_pro_1_arr_all
            valid_y_pro_0_arr_all_f[key] = valid_y_pro_0_arr_all
    
    
    Ensemble_Classifier_Result(valid_y_class_arr_all_f, valid_y_pro_1_arr_all_f, valid_y_pro_0_arr_all_f, model_weight_all, train_labels, valids, "soft")
    Ensemble_Classifier_Result(valid_y_class_arr_all_f, valid_y_pro_1_arr_all_f, valid_y_pro_0_arr_all_f, model_weight_all, train_labels, valids, "hard")
    Ensemble_Classifier_Result(valid_y_class_arr_all_f, valid_y_pro_1_arr_all_f, valid_y_pro_0_arr_all_f, model_weight_all, train_labels, valids, "devs", 3, 0.05)

    # # 测试
    # print('hard:')
    # Ensemble_Classifier_Test(train_features_fs, train_labels, test_x, test_y, index_arr, model_obj_arr, 'hard', 100000, 100000)
    # print('soft:')
    # Ensemble_Classifier_Test(train_features_fs, train_labels, test_x, test_y, index_arr, model_obj_arr, 'soft', 100000, 100000)
    # print('devs:')
    # Ensemble_Classifier_Test(train_features_fs, train_labels, test_x, test_y, index_arr, model_obj_arr, 'devs', 3, 0.05)

    # 预测
    # print('devs:')
    # Ensemble_Classifier_Pred(train_features_fs, train_labels, test_x, index_arr, model_obj_arr, 'devs', 3, 0.05)

    print("*************************************************************************************************************************************")

    print(time.time() - start)
    print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
