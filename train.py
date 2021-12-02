import os 
import pandas as pd

import torch
import torch.nn as nn

from model import *
from tqdm import tqdm
import model_decoder
from datetime import datetime
import argparse


import sklearn.metrics
from statistics import mean
from utils import *
import logging
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter


#TODO: implement scheduler 
# CUDA_LAUNCH_BLOCKING=1


best_test_f1 = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################
#arg parser
########################################################################

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name', required=True)
parser.add_argument('--group_num', required=True)
parser.add_argument('--time_step', required=True)
parser.add_argument('--role', required=True)
parser.add_argument('--ablation', default = None)
parser.add_argument('--epochs', default = 250)
parser.add_argument('--init_seed', required=True)
parser.add_argument('--cv_seed', required=True)
parser.add_argument('--task', default = "next_speaker")

args = parser.parse_args()
model_name = args.model_name
group_num = int(args.group_num)
time_step = int(args.time_step)
role = int(args.role)
ablation = args.ablation
init_seed = int(args.init_seed)
cv_seed = int(args.cv_seed)
task = str(args.task)

if task == 'next_speaker':
    from dataset_slide import *
elif task == 'identify_speaker':
    from dataset_slide_identify import *
else:
    print("task not specified choose from [next_speaker, identify_speaker] ")
    exit()





torch.manual_seed(init_seed)

########################################################################
#ablations
########################################################################


behavior_types = {
    'Acknowledgement' : [*range(1,7)],
    'Body' : [*range(7,14)],
    'Head' : [*range(14, 20)],
    'Hand' : [*range(20,22)],
    'Eye' : [*range(22,25)],
    'Face' : [*range(25,30)],
    'Positive_Verbal' : [*range(30,35)],
    'Negative_Verbal' : [*range(35,39)],
    'all_nonverbal' : [*range(7, 30)]
    } 

input_feats = 38

if ablation == 'all_verbal':
    abl_range = ((np.array(behavior_types['Acknowledgement']) - 1),(np.array(behavior_types['Positive_Verbal']) - 1),(np.array(behavior_types['Negative_Verbal'])  - 1))
    abl_range = np.concatenate(abl_range)
    input_feats -= len(abl_range)
    

elif ablation:
    input_feats -= len(behavior_types[ablation])
    abl_range = (behavior_types[ablation][0]-1, behavior_types[ablation][-1])

########################################################################
#model selection
########################################################################

if "Graph" in model_name:
    if role == 1:
        model = model_decoder.Decoder_Role(num_class = group_num, time_step = time_step, state_dim = input_feats).to(device)

    if role == 0: 
        model = model_decoder.Decoder_NoRole(num_class = group_num, time_step = time_step, state_dim = input_feats).to(device)
else:
    model = ClassifierSmall(group_num, input_feats=input_feats * group_num, out_feats = group_num + 1).to(device)

# if model_name == "RIG_dong":
#     model = relRIG.RIG(num_class = group_num, time_step = 0).to(device)
# elif model_name == "RIG_no_rel":
#     model = norelRIG.RIG(num_class = group_num, time_step = 0).to(device)
# elif model_name == "RIG_dong_3":
#     model = relRIG.RIG(num_class = group_num, time_step = 3).to(device)
# elif model_name == "RIG_no_rel_3":
#     model = norelRIG.RIG(num_class = group_num, time_step = 3).to(device)


########################################################################
#model logging
########################################################################

print("Model Chosen: {}".format(model.__class__.__name__))

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
model_unique = "{model}".format(model = model_name + "_timestep{}".format(time_step) + "_role{}".format(role)+ "_ablation{}".format(ablation) + "_init_seed{}".format(init_seed) + "_cv_seed{}".format(cv_seed))
writer = SummaryWriter("./runs/{model}_{dt_string}".format(model = model_unique, dt_string = dt_string))

weight_dir = "./logs/"
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

log_dir = "./logs/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = log_dir + "{model}.log".format(model = model_unique)
logging.basicConfig(filename=log_path)


########################################################################
#Group Num Specific Models
########################################################################
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

#CV - fix seed for numpy only
np.random.seed(init_seed)
index_list = np.arange(len(SD))
np.random.shuffle(index_list)
test_range = index_list[list(range(test_len*(cv_seed), test_len*(cv_seed+1)))]
train_range = index_list[list(set(range(len(SD))) - set(test_range))]

train = torch.utils.data.Subset(SD, train_range)
test = torch.utils.data.Subset(SD, test_range)
# train, test = torch.utils.data.random_split(SD, (train_len, test_len), generator=torch.Generator().manual_seed(seed))

batch_size = 32
epochs = int(args.epochs)
trainloader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 8)
testloader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = 8)

########################################################################
#optimizer and lr scheduler and early stopping - Not Used For Now 
########################################################################


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
class_weights = torch.tensor([0.2858, 7.0487, 3.9988, 9.1401]).to(device)

if group_num == 3:
    class_weights = torch.tensor([0.2858, 7.0487, 3.9988, 9.1401]).to(device)

elif group_num == 4:
    class_weights = torch.tensor([0.2841, 3.6039, 5.2514, 1.2944, 4.1773]).to(device)

else:
    print("This number of group members is not supported!")
    print(exit())


if task == 'identify_speaker':
    loss_fn = nn.CrossEntropyLoss()
else:
    loss_fn = nn.CrossEntropyLoss(weight = class_weights, reduction = 'mean')

# print('Initializing learning rate scheduler...')
# lr_scheduler = LRScheduler(optimizer)

# print('Initializing early stopping...')
# early_stopping = EarlyStopping()

########################################################################
#training
########################################################################


for epoch in range(epochs):
    print('EPOCH: {}'.format(epoch))
    losses = []
    accur = []
    prop = []
    f1 = []
    overlap_c = 0
    for j,batch in enumerate(tqdm(trainloader)):
        social_rel = batch['social_rel']
        overlap = int(batch['overlap'].sum())
        overlap_c += overlap
        
        

        context, vb_output = batch['context'], batch['vb_output']

        if ablation == 'all_verbal':
            desired_ind = list(set(list(range(38))) - set(abl_range))
            context = context[...,desired_ind]
            
        
        elif ablation =='all_nonverbal':
            desired_ind = list(set(list(range(38))) - set(np.array(behavior_types['all_nonverbal']) -1))
            context = context[...,desired_ind]
        

        elif ablation:
            desired_ind = list(set(list(range(38))) - set(list(range(abl_range[0], abl_range[1]))))
            context = context[...,desired_ind]

        if "Classifier" in model_name:
            x_train = context.to(device).flatten(start_dim =1)
        else:
            x_train = context.to(device)
        
        labels = vb_output.sum(2).to(device).flatten(start_dim =1)
        index_labels = torch.zeros(x_train.shape[0]).long().to(device)
        index_labels[labels.nonzero()[:,0]] = labels.nonzero()[:,1] + 1 
        y_train = index_labels


        optimizer.zero_grad()

        output = model(x_train, social_rel)

        
        loss = loss_fn(output.squeeze(), y_train.long())
        predictions = torch.argmax(output, dim = 1)
        acc = torch.mean((predictions == y_train).float())
        f1_score = sklearn.metrics.f1_score(predictions.cpu().numpy(), y_train.cpu().numpy(), average='macro')

        losses.append(loss.detach())
        accur.append(acc.detach())
        f1.append(f1_score)

        
        #accuracy

        loss.backward()
        optimizer.step()


        
    
    with torch.set_grad_enabled(False):
        test_losses = []
        test_accur = []
        test_prop = []
        test_f1 = []
        test_f1_weighted = []
        confusion_mat = np.zeros((4,4))
        for i,batch in enumerate(tqdm(testloader)):
            social_rel = batch['social_rel']
            
            context, vb_output = batch['context'], batch['vb_output']


            if ablation == 'all_verbal':
                desired_ind = list(set(list(range(38))) - set(abl_range))
                context = context[...,desired_ind]

            elif ablation =='all_nonverbal':
                desired_ind = list(set(list(range(38))) - set(np.array(behavior_types['all_nonverbal']) -1))
                context = context[...,desired_ind]

            
            elif ablation:
                
                desired_ind = list(set(list(range(38))) - set(list(range(abl_range[0], abl_range[1]))))
                context = context[...,desired_ind]

            if "Classifier" in model_name:
                x_test = context.to(device).flatten(start_dim =1)
            else:
                x_test = context.to(device)
            labels = vb_output.sum(2).to(device).flatten(start_dim =1)
            index_labels = torch.zeros(x_test.shape[0]).long().to(device)
            index_labels[labels.nonzero()[:,0]] = labels.nonzero()[:,1] + 1 
            y_test = index_labels
            output = model(x_test, social_rel)
            test_loss = loss_fn(output.squeeze(), y_test.long())
            predictions = torch.argmax(output, dim = 1)

            # auprc = sklearn.metrics.average_precision_score(output.squeeze().cpu(), y_test.long().cpu())
            test_f1_score = sklearn.metrics.f1_score(predictions.cpu().numpy(), y_test.cpu().numpy(), average='macro')
            # print(auprc)
    
            test_f1_score_weighted = sklearn.metrics.f1_score(predictions.cpu().numpy(), y_test.cpu().numpy(), average='weighted')

            test_acc = torch.mean((predictions == y_test).float())

            test_losses.append(test_loss.detach())
            test_accur.append(test_acc.detach())
            test_f1.append(test_f1_score)
            test_f1_weighted.append(test_f1_score_weighted)

            try:
                confusion_mat = confusion_mat + confusion_matrix(y_test.detach().cpu().numpy(), predictions.detach().cpu().numpy(), labels=[0,1,2,3])
            except Exception:
                pdb.set_trace()

    

            


        epoch_test_loss = torch.mean(torch.stack(test_losses))
        epoch_test_acc = torch.mean(torch.stack(test_accur))
        epoch_test_f1 = mean(test_f1)
        epoch_test_f1_weighted = mean(test_f1_weighted)

 

        # lr_scheduler(epoch_val_f1)
        # early_stopping(epoch_val_loss)
        # if early_stopping.early_stop:
        #     break

            
    
    
    epoch_loss = torch.mean(torch.stack(losses))
    epoch_acc = torch.mean(torch.stack(accur))
    epoch_f1 = mean(f1)
    
    print("acc:", epoch_acc)
    print("f1:", epoch_f1)
    print("loss:", epoch_loss)


    print("test_acc:", epoch_test_acc)
    print("test_f1:", epoch_test_f1)
    print("test_loss:", epoch_test_loss)


    writer.add_scalar("Accuracy/train", epoch_acc, epoch)

    writer.add_scalar("Accuracy/test", epoch_test_acc, epoch)

    writer.add_scalar("Loss/train", epoch_loss, epoch)

    writer.add_scalar("Loss/test", epoch_test_loss, epoch)

    writer.add_scalar("f1/train", epoch_f1, epoch)

    writer.add_scalar("f1/test", epoch_test_f1, epoch)

    
    if best_test_f1 < epoch_test_f1: 
        best_test_f1 =  epoch_test_f1
        path = weight_dir + "{model}.pth".format(model = model_unique)

        torch.save(model.state_dict(), path)

        log_statement = "Best model at Epoch: {epoch}, acc: {acc}, loss: {loss}, f1:{f1}, weighted_f1:{weighted_f1}".format( epoch = epoch, acc = epoch_test_acc, loss = epoch_test_loss, f1 = epoch_test_f1, weighted_f1 = epoch_test_f1_weighted)
        logging.warning(log_statement)

