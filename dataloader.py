import os
import torch
from torch.utils import data
import numpy as np
import pickle
import random
import json

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, data_type="train"):
        self.data = data
        self.len = len(self.data)
        # self.subjects_dict = subjects_dict
        self.data_type = data_type
        # self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""

        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]

        return torch.FloatTensor(audio),torch.FloatTensor(vertice), file_name

    def __len__(self):
        return self.len


def get_metadata(blendshape_path, data_path, arch, device, train_list=None):

    if blendshape_path is not None:
        #test
        with open(blendshape_path, 'rb') as f:
            blendshape_dict = pickle.load(f)

    data = []
    
    for wav_path in os.listdir(data_path):
        # remove some error code 
        if blendshape_path is not None and 'train' in blendshape_path:
            # generate small train set
            if wav_path.split(".")[0] not in train_list:
                continue
        fileshort_name = wav_path.split(".")[0]
        file_meta_dict = dict()
        if arch == 'mfcc':
            npy_disk_path = os.path.join(data_path, wav_path)
            assert os.path.exists(npy_disk_path)
            input_values = np.load(npy_disk_path, allow_pickle=True)
            file_meta_dict["audio"] = input_values
            file_meta_dict["name"] = fileshort_name
        elif arch == 'ppg':
            npy_disk_path = os.path.join(data_path, wav_path)
            assert os.path.exists(npy_disk_path)
            input_values = torch.load(npy_disk_path,map_location='cpu')
            file_meta_dict["audio"] = input_values
            file_meta_dict["name"] = fileshort_name
        elif arch in [ 'chinese_wav2vec2_base', 'chinese_wav2vec2_large']:
            npy_disk_path = os.path.join(data_path, wav_path)
            assert os.path.exists(npy_disk_path)
            input_values = torch.load(npy_disk_path,map_location='cpu')
            input_values_new = [ i[0] for i in input_values]
            input_values = torch.stack(input_values_new).squeeze(2) #12, 251, 768 
            input_values = input_values.type(torch.FloatTensor)
            file_meta_dict["audio"] = input_values
            file_meta_dict["name"] = fileshort_name #torch.float16
        elif arch in [ 'chinese_hubert_base', 'chinese_hubert_large']:
            npy_disk_path = os.path.join(data_path, wav_path)
            assert os.path.exists(npy_disk_path)
            input_values = torch.load(npy_disk_path,map_location='cpu')
            # pdb.set_trace()
            input_values_new = [ i[0] for i in input_values]
            input_values = torch.stack(input_values_new).squeeze(2) #12, 251, 768 
            input_values = input_values.type(torch.FloatTensor)
            file_meta_dict["audio"] = input_values
            file_meta_dict["name"] = fileshort_name #torch.float16
        else:
            print('check arch type!')
            exit(1)

        if blendshape_path is not None:
            if fileshort_name in blendshape_dict.keys():
                file_meta_dict["vertice"] = blendshape_dict[fileshort_name]
            # file_meta_dict["vertice"] = np.zeros((1,52))
        else:
            # test, average shenxiaoya todo current 0
            file_meta_dict["vertice"] = np.zeros((1,52)) + 0.00001
        data.append(file_meta_dict)
    return data

def read_data(args):
    print("Loading data...")
    
    random.seed(args.seed)
    train_meta_list = []
    with open(args.train_json,'r') as f:
        train_list = json.load(f)
    for speaker in args.train_speaker_list:
        train_data_path = os.path.join(args.root_dir,'wav_features', speaker, args.feature_dir, 'train')
        train_meta_list += get_metadata(args.train_blendshape_path, train_data_path, args.arch, args.device,train_list)
    val_data_path = os.path.join(args.root_dir,'wav_features', 'org', args.feature_dir, 'val_1')
    val_meta_list = get_metadata(args.val_blendshape_path, val_data_path, args.arch, args.device)
    
    if args.output_path == 'result':
        # output of data in org test set
        test_data_path = os.path.join(args.root_dir,'wav_features', 'org', args.feature_dir, 'test_a')
        test_meta_list = get_metadata(None, test_data_path, args.arch, args.device)

    else:
        # output of data in wild dataset
        test_data_path =  args.test_input_path
        test_meta_list = get_metadata(None, test_data_path, args.arch, args.device)
    
    print('{} sequences in train set; {} sequences in val set; {} sequences in test set'.format(len(train_meta_list), len(val_meta_list), len(test_meta_list)))
    return train_meta_list, val_meta_list, test_meta_list

# modified from https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
# only used in hubert/wav2vec2 (#layers,T,feature_dim)
# can not use in mfcc(T,feature_dim)
def collate_fn1(data):   

    audio_real_len_list = list(i[0].shape[1] for i in data) # i[0] shape is (num of layer, audio_len, 1024)
    audio_max_len = max(audio_real_len_list) # i[1] shape is (blendshape len, 37) audio_len is amost 2x than blendshape len
    audio_file_names = list(i[2] for i in data)
    
    blendshape_real_len_list = list(i[1].shape[0] for i in data)
    blendshape_max_len = max(blendshape_real_len_list)

    audio_features = torch.zeros((len(data), data[0][0].shape[0], audio_max_len, data[0][0].shape[2]))
    blendshape_labels = torch.zeros((len(data), blendshape_max_len, data[0][1].shape[1]))

    for i in range(len(data)):
        example = data[i]
        audio_expand_len, blendshape_expand_len = example[0].shape[1], example[1].shape[0]
        audio_features[i] = torch.cat([example[0], torch.zeros((example[0].shape[0], audio_max_len - audio_expand_len, example[0].shape[2]))],axis=1)
        blendshape_labels[i] = torch.cat([example[1], torch.zeros((blendshape_max_len - blendshape_expand_len, example[1].shape[1]))],axis=0)

    audio_lengths = torch.tensor(audio_real_len_list)
    blendshape_lengths = torch.tensor(blendshape_real_len_list)

    return audio_features.float(), blendshape_labels.float(), audio_lengths.long(), blendshape_lengths.long(), audio_file_names

# can be used in mfcc feature (T,feature_dim)
def collate_fn2(data):   

    audio_real_len_list = list(i[0].shape[0] for i in data) # i[0] shape is (audio_len, feature_dim)
    audio_max_len = max(audio_real_len_list) 
    audio_file_names = list(i[2] for i in data)
    
    blendshape_real_len_list = list(i[1].shape[0] for i in data) # i[1] shape is (blendshape len, 37) audio_len is amost 2x than blendshape len
    blendshape_max_len = max(blendshape_real_len_list)

    audio_features = torch.zeros((len(data), audio_max_len, data[0][0].shape[1]))
    blendshape_labels = torch.zeros((len(data), blendshape_max_len, data[0][1].shape[1]))

    for i in range(len(data)):
        example = data[i]
        audio_expand_len, blendshape_expand_len = example[0].shape[0], example[1].shape[0]
        audio_features[i] = torch.cat([example[0], torch.zeros((audio_max_len - audio_expand_len, example[0].shape[1]))],axis=0)
        blendshape_labels[i] = torch.cat([example[1], torch.zeros((blendshape_max_len - blendshape_expand_len, example[1].shape[1]))],axis=0)

    audio_lengths = torch.tensor(audio_real_len_list)
    blendshape_lengths = torch.tensor(blendshape_real_len_list)

    return audio_features.float(), blendshape_labels.float(), audio_lengths.long(), blendshape_lengths.long(), audio_file_names

def get_dataloaders(args):
    dataset = dict()
    train_data, valid_data, test_data = read_data(args)
    train_data = Dataset(train_data,"train")
    valid_data = Dataset(valid_data,"val")
    test_data = Dataset(test_data,"test")
    if args.arch in ["chinese_hubert_large","chinese_hubert_base","chinese_wav2vec2_large","chinese_wav2vec2_base"]:
        collate_fn = collate_fn1
    elif args.arch in ['mfcc','ppg']:
        collate_fn = collate_fn2      
    else:
        print('please assign collate_fn for arch {}'.format(args.arch))
        
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers = args.num_workers)   
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate_fn, num_workers = args.num_workers)    
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, num_workers = args.num_workers)
        
    return dataset
