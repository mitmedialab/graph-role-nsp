from dataset_slide import *
import torch
import torch.nn as nn

import numpy as np

from scipy.stats import uniform, randint

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import xgboost as xgb

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
import sklearn
import logging
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--group_num', required=True)
parser.add_argument('--task', default = "next_speaker")
args = parser.parse_args()
group_num = int(args.group_num)
task = str(args.task)


if task == 'next_speaker':
    from dataset_slide import *
elif task == 'identify_speaker':
    from dataset_slide_identify import *
else:
    print("task not specified choose from [next_speaker, identify_speaker] ")
    exit()



def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X, y)]
    
    clf.fit(X, y,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X)
    accuracy = accuracy_score(y, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

person_order = {'F1_Interaction_1': {'P2': 1, 'P1': 1, 'P3': 2},
 'F1_Interaction_2': {'P2': 1, 'P1': 1, 'P3': 2},
 'F2_Interaction_1': {'P4': 1, 'P5': 3},
 'F2_Interaction_2': {'P4': 1},
 'F3_Interaction_1': {'P8': 3, 'P6': 1, 'P7': 1},
 'F3_Interaction_2': {'P6': 1, 'P7': 1},
 'F4_Interaction_1': {'P14': 2,
  'P12': 1,
  'P11': 1,
  'P10': 1,
  'P9': 1,
  'P13': 3},
 'F4_Interaction_2': {'P12': 1,
  'P11': 1,
  'P10': 1,
  'P9': 1,
  'P13': 3},
 'F5_Interaction_1': {'P16': 2, 'P15': 1},
 'F5_Interaction_2': {'P16': 2, 'P15': 1},
 'F6_Interaction_1': {'P19': 3, 'P18': 1, 'P17': 1},
 'F6_Interaction_2': {'P19': 3, 'P18': 1, 'P17': 1},
 'F7_Interaction_1': {'P22': 3,
  'P20': 1,
  'P21': 1,
  'P23': 2},
 'F8_Interaction_1': {'P24': 1, 'P25': 3},
 'F8_Interaction_2': {'P24': 1, 'P25': 3},
 'F8_Interaction_3': {'P24': 1, 'P25': 3},
 'F10_Interaction_1': {'P27': 1, 'P28': 1},
 'F11_Interaction_1': {'P29': 1, 'P30': 2},
 'F11_Interaction_2': {'P29': 1, 'P30': 2},
 'F13_Interaction_1': {'P32': 1, 'P33': 2},
 'F17_Interaction_1': {'P37': 1, 'P38': 2},
 'F17_Interaction_2': {'P37': 1, 'P38': 2}}


group_nums = {1: ['F2_Interaction_2'],
 2: ['F2_Interaction_1',
  'F3_Interaction_2',
  'F5_Interaction_1',
  'F5_Interaction_2',
  'F8_Interaction_1',
  'F8_Interaction_2',
  'F8_Interaction_3',
  'F10_Interaction_1',
  'F11_Interaction_1',
  'F11_Interaction_2',
  'F13_Interaction_1',
  'F17_Interaction_1',
  'F17_Interaction_2'],
 3: ['F1_Interaction_1',
  'F1_Interaction_2',
  'F3_Interaction_1',
  'F6_Interaction_1',
  'F6_Interaction_2'],
 4: ['F7_Interaction_1'],
 5: ['F4_Interaction_2'],
 6: ['F4_Interaction_1']}

group_all_dataset = []
group_ids = group_nums[group_num]
for group_id in group_ids:
    group_specific_dataset = SpeedDatingDS(group_id = group_id, social_rel = person_order[group_id])
    group_all_dataset.append(group_specific_dataset)

SD = torch.utils.data.ConcatDataset(group_all_dataset)

########################################################################
#Dataloader
########################################################################
train_len = len(SD) - len(SD)//5
test_len = len(SD)//5

for init_seed in [2]:
    for cv_seed in [0,1,2,3,4]:


        np.random.seed(init_seed)
        index_list = np.arange(len(SD))
        np.random.shuffle(index_list)
        test_range = index_list[list(range(test_len*(cv_seed), test_len*(cv_seed+1)))]
        train_range = index_list[list(set(range(len(SD))) - set(test_range))]
        train = torch.utils.data.Subset(SD, train_range)
        test = torch.utils.data.Subset(SD, test_range)

        batch_size = 32
        trainloader = DataLoader(train, batch_size = train_len, shuffle = True, num_workers = 2)
        testloader = DataLoader(test, batch_size = test_len, shuffle = True, num_workers = 2)

        device = "cpu"

        for idx, batch in enumerate(trainloader):

            x_train, vb_output = batch['context'], batch['vb_output']

            labels = vb_output.sum(2).to(device).flatten(start_dim =1)
            index_labels = torch.zeros(x_train.shape[0]).long().to(device)
            index_labels[labels.nonzero()[:,0]] = labels.nonzero()[:,1] + 1 
            y_train = index_labels

        for idx, batch in enumerate(testloader):
            x_test, vb_output = batch['context'], batch['vb_output']

            labels = vb_output.sum(2).to(device).flatten(start_dim =1)
            index_labels = torch.zeros(x_test.shape[0]).long().to(device)
            index_labels[labels.nonzero()[:,0]] = labels.nonzero()[:,1] + 1 
            y_test = index_labels

        X = x_train.flatten(start_dim =1).cpu().numpy()
        y = y_train.cpu().numpy()

        x_test = x_test.flatten(start_dim =1).cpu().numpy()
        y_test = y_test.cpu().numpy()

        space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
        }

        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                                space = space,
                                algo = tpe.suggest,
                                max_evals = 10,
                                trials = trials)
   
        clf = xgb.XGBClassifier(best_hyperparams)

        clf.fit(X, y)
        y_true, y_pred = y_test, clf.predict(x_test)

        f1 = sklearn.metrics.f1_score(y_pred, y_true, average='macro')
        acc = (y_pred == y_true).mean()     
    #     print("f1: {}".format()))
    #     print("weighted_f1: {}".format(sklearn.metrics.f1_score(y_pred, y_true, average='weighted')))
    #     print("acc: {}".format((y_pred == y_true).mean()))
    #     print(confusion_matrix(y_test, y_pred))

        model_unique = "{model}".format(model = "XGBoost_init_seed{}".format(init_seed) + "_cv_seed{}".format(cv_seed))

        weight_dir = "./model_weights/"
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        log_dir = "./logs/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        clf.save_model("./model_weights/{}.json".format(model_unique))


        log_path = log_dir + "{model}.log".format(model = model_unique)
        print(f1)
        with open(log_path, 'w') as f:
            f.write('\n')
            f.write("{}".format(model_unique))
            f.write('\n')
            f.write("acc: {acc}, f1:{f1} ".format(acc = acc, f1 = f1))
            
            