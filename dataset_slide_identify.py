import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os 
import pandas as pd
import pdb

random.seed(0)

#cuda utils

#create sliding window -> start to end 


class SpeedDatingDS(Dataset):
    def __init__(self, group_id, social_rel, context_len = 5, hop = 1, dir_path = "./SpeedDating/", transform=None):
        #other hyperparms
        self.fps = 30
        self.context_len = context_len * self.fps
        self.hop = hop

        #load csv 
        nvb_csv_list = []
        for path, currentDirectory, files in os.walk(os.path.join(dir_path, 'nonverbal')):
            for file in files:
                if file.startswith(group_id):
                    nvb_csv_list.append(os.path.join(path, file))
        

        nvb_dfs = []
        vb_dfs = []
        for csv_path in nvb_csv_list:  
            
            nvb_dfs.append(pd.read_csv(csv_path))
            vb_dfs.append(pd.read_csv(csv_path.replace('nonverbal', 'verbal')))

        
        self.nvb_df = pd.concat(nvb_dfs)
        self.nvb_df = self.nvb_df.loc[:, ['Start_frame', 'End_frame', 'Participant', 'Behavior_ID']]

        self.vb_df = pd.concat(vb_dfs)
        self.vb_df = self.vb_df.sort_values(by = ['End_frame'])
        self.vb_df = self.vb_df.loc[:, ['Start_frame', 'End_frame', 'Participant', 'Behavior_ID']]

        self.vb_df["Start_frame"] = round(self.vb_df["Start_frame"]*self.fps)
        self.vb_df["End_frame"] = round(self.vb_df["End_frame"]*self.fps)
        
        #finding num_nodes - needs to be fixed in the future

        self.num_ppl = len(self.nvb_df["Participant"].unique())

        self.num_context = 38


        self.unique_nvb = np.unique(self.nvb_df["Behavior_ID"].to_numpy())
        self.unique_vb = np.unique(self.vb_df["Behavior_ID"].to_numpy())
        self.unique_person = np.unique(self.nvb_df["Participant"].to_numpy())

        #social relations
        self.social_rel = social_rel

        self.social_rel_vec = torch.zeros(self.num_ppl)
        for i,person in enumerate(self.unique_person):
            self.social_rel_vec[i] = social_rel[person]

            

    
        
        
    def __len__(self):
        return len(self.vb_df)

        
    def __getitem__(self, idx):
        

        vb_start_frame  = int(self.vb_df.iloc[idx]['Start_frame'])

        curr_context_nvb = self.nvb_df[self.nvb_df['Start_frame'].le(vb_start_frame)]
        curr_context_nvb = curr_context_nvb[curr_context_nvb['End_frame'].ge(vb_start_frame - self.context_len)]

        curr_context_vb = self.vb_df[self.vb_df['Start_frame'].le(vb_start_frame)]
        curr_context_vb = curr_context_vb[curr_context_vb['End_frame'].ge(vb_start_frame - self.context_len)]

        #lets represent each input modality as one-hot vectors
        

        context = torch.zeros(self.num_ppl, self.num_context)
        
        for i, speaker in enumerate(self.unique_person):
            speaker_series = curr_context_nvb[curr_context_nvb['Participant'] == speaker]
            if not speaker_series.empty:
                
                nvb_idx = speaker_series.tail(1)
                nvb_idx = int(nvb_idx['Behavior_ID'])
                context[i, nvb_idx -1] = 1

        for i, speaker in enumerate(self.unique_person):
            speaker_series = curr_context_vb[curr_context_vb['Participant'] == speaker]
            if not speaker_series.empty:
                vb_idx = speaker_series.tail(1)
                vb_idx = int(vb_idx['Behavior_ID'])
                context[i, vb_idx -1 ] = 1


        
        #lets represent each output modality as one-hot vectors
        overlap = torch.zeros(1)


        vb_output = torch.zeros(self.num_ppl, self.num_context)

        person_id = self.vb_df.iloc[idx]['Participant']
        person_id =  np.where(self.unique_person == person_id)
        vb_idx = int(self.vb_df.iloc[idx]['Behavior_ID'])
        #vb_idx = np.where(self.unique_vb == vb_idx)
        vb_output[person_id,vb_idx -1] = 1


        # output_label = self.vb_df[self.vb_df['Start_frame'].le(speaking_time)]
        # output_label = output_label[output_label['End_frame'].ge(speaking_time)]
        # if not output_label.empty:
        #     if len(output_label) > 1:
        #         #print("Overlap!")
        #         overlap = torch.ones(1)
        #     try:
        #         person_id = output_label['Participant'].tail(1).item()
        #         person_id =  np.where(self.unique_person == person_id)
        #         vb_idx = int(output_label['Behavior_ID'].tail(1).item())
        #         if vb_idx in [2, 31]:
        #             pass
        #         else:
        #             vb_output[person_id,vb_idx -1] = 1

        #     except Exception:
        #         pdb.set_trace()
        
        #vb_idx = np.where(self.unique_vb == vb_idx)
       
        

        #sample non-vb regions
        
        sample = {

            'context': context,
            'vb_output': vb_output,
            'social_rel':  self.social_rel_vec,
            'overlap' : overlap

        }
        return sample
        
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


print("Loading...")
context_len = 5
group_num = 3
group_all_dataset = []
group_ids = group_nums[group_num]
for group_id in group_ids:
    group_specific_dataset = SpeedDatingDS(group_id, person_order[group_id], context_len)
    group_all_dataset.append(group_specific_dataset)

SD = torch.utils.data.ConcatDataset(group_all_dataset)

print("Finished Loading...")

turnloader = DataLoader(SD, batch_size = 179, shuffle = True)

for idx, batch in enumerate(turnloader):

    x_turn, vb_output, social_rel = batch['context'], batch['vb_output'], batch['social_rel']
    labels = vb_output.sum(2).flatten(start_dim =1)
    index_labels = torch.zeros(x_turn.shape[0]).long()
    index_labels[labels.nonzero()[:,0]] = labels.nonzero()[:,1] + 1 
    y_turn = index_labels