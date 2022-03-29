import torch.utils.data as data
from PIL import Image
import os
import os.path
from copy import copy
import numpy as np
import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Union
from PATH import PATH
PRESET_VARS = PATH()
import pickle
import pandas as pd
from utils.data_utils import extact_face_landmarks

class DatasetBase(data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

    def _get_all_label(self):
        raise NotImplementedError
    

class DataModuleBase(pl.LightningDataModule):
    def __init__(self, 
        num_workers_train:int = 0, num_workers_test:int = 0):

        self.num_workers_train = num_workers_train
        self.num_workers_test = num_workers_test

    def setup(self, stage = None):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

class ImageDataset(DatasetBase):
    def __init__(self, mode:str, transforms_train=None, transforms_test=None, parse_label_func=lambda x:x, annotation_file=None):
        """Summary
        
        Args:
            mode (str): the train set, val set ot test set
            transforms_train (None, optional): transformation function
            transforms_test (None, optional): transformation function
            parse_label_func (TYPE, optional): the function to parse the emotion labels from dataframe
            annotation_file (Optional[str], optional): annotation file path
        """
        self.mode = mode  
        assert self.mode in ['Train', 'Validation', 'Test'], "Wrong mode!"
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test
        self.annotation_file = annotation_file
        self.read_annotaion_file()
        self.parse_label_func = parse_label_func
    def read_annotaion_file(self):
        if isinstance(self.annotation_file, str):
            data = pickle.load(open(self.annotation_file, 'rb'))
        elif isinstance(self.annotation_file, dict):
            data = self.annotation_file
        data = data[self.mode]
        if self.mode =='Train':
            self._transform = self.transforms_train
        else:
            self._transform = self.transforms_test
        self.df = self.concatenate_videos_to_one_df(data)

    def concatenate_videos_to_one_df(self, data_dict):
        videos = list(data_dict.keys())
        keys = data_dict[videos[0]].keys()
        dfs = []
        for video in videos:
            assert (data_dict[video].keys() == keys).all(), "Failed to concatenate multiple dfs because of the keys mismatch!"
            dfs.append(data_dict[video])
        return pd.concat(dfs, ignore_index=True)

    def __getitem__(self,
        index:int):
        row = self.df.iloc[index]
        label = self.parse_label_func(row)
        img_path = row['path']
        if not os.path.exists(img_path):
            # replace the directory 
            dirname = "/media/Samsung/ABAW3_Challenge"
            if dirname in img_path:
                img_path = img_path.replace(dirname, "../../scratch")
                if not os.path.exists(img_path):
                    USERDIR = os.environ.get('TACC_USERDIR')
                    img_path = img_path.replace("../../scratch", os.path.join(USERDIR, 'personal_datasets'))

        image = Image.open(img_path).convert("RGB")
        image = self._transform(image)
        return image, label

    def _get_all_label(self):
        return self.parse_label_func(self.df) 
    @property
    def ids(self):
        return np.arange(len(self.df))
    @property
    def dataset_size(self):
        return len(self.ids)
    def __len__(self):
        return self.dataset_size

class ImageSequenceDataset(DatasetBase):
    def __init__(self, mode:str, seq_len: int, transforms_train=None, transforms_test=None, parse_label_func=lambda x:x, annotation_file=None):
        self.mode = mode  
        assert self.mode in ['Train', 'Validation', 'Test'], "Wrong mode!"
        self.seq_len = seq_len
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test
        self.annotation_file = annotation_file
        self.read_annotaion_file()
        self.parse_label_func = parse_label_func
    def read_annotaion_file(self):
        if isinstance(self.annotation_file, str):
            data = pickle.load(open(self.annotation_file, 'rb'))
        elif isinstance(self.annotation_file, dict):
            data = self.annotation_file
        data = data[self.mode]
        if self.mode =='Train':
            self._transform = self.transforms_train
        else:
            self._transform = self.transforms_test
        self.df, self.sample_seqs = self.concatenate_videos_sample_sequence(data)
    def concatenate_videos_sample_sequence(self, data_dict):
        videos = list(data_dict.keys())
        keys = data_dict[videos[0]].keys()
        dfs = []
        sample_seqs = []
        N = 0
        for video in videos:
            assert (data_dict[video].keys() == keys).all(), "Failed to concatenate multiple dfs because of the keys mismatch!"
            video_df = data_dict[video]
            length = len(video_df)
            if length//self.seq_len ==0:
                continue
                print('one video filtered out because of its short length')
                print(video_df)
            for i in range(length//self.seq_len + 1):
                if i !=length//self.seq_len:
                    start, end = i*self.seq_len, (i+1)*self.seq_len
                else:
                    start, end = length - self.seq_len, length
                sample_seqs.append([N+start, N+end])
            N+=length
            dfs.append(video_df)
        return pd.concat(dfs, ignore_index=True), sample_seqs
    
    def __getitem__(self,
        index:int):
        start, end = self.sample_seqs[index]
        df = self.df.iloc[start:end]
        images, labels = [], []
        image_paths = []
        for _, row in df.iterrows():
            label = self.parse_label_func(row)
            img_path = row['path']
            if not os.path.exists(img_path):
                # replace the directory 
                dirname = "/media/Samsung/ABAW3_Challenge"
                if dirname in img_path:
                    img_path = img_path.replace(dirname, "../../scratch")
                    if not os.path.exists(img_path):
                        USERDIR = os.environ.get('TACC_USERDIR')
                        img_path = img_path.replace("../../scratch", os.path.join(USERDIR, 'personal_datasets'))

            image = Image.open(img_path).convert("RGB")
            image = self._transform(image)
            images.append(image)
            labels.append(label)
            image_paths.append(row['path'])
        labels = np.stack(labels, axis=0)
        return torch.stack(images, dim=0), torch.from_numpy(labels), image_paths
    @property
    def ids(self):
        return np.arange(len(self.sample_seqs))
    @property
    def dataset_size(self):
        return len(self.ids)
    def __len__(self):
        return self.dataset_size
    def _get_all_label(self):
        return self.parse_label_func(self.df) 

class CV_Dataset(DatasetBase):
    def __init__(self, mode: str, 
        train_ids:Union[int, List[int]]=None, val_id:Optional[int]=None, 
        transforms_train=None, transforms_test=None, 
        parse_label_func=lambda x:x, parse_attention_func=None, 
        annotation_file:Optional[str]=None, 
        n_folds:int=3,
        keys_partition = None):
        """Summary
        
        Args:
            mode (str): the train set, val set ot test set
            train_ids (Union[int, List[int]], optional): specifying the indexes of train set part in the N folds
            val_id (Optional[int], optional): specifying the index of val set part in the N folds
            transforms_train (None, optional): transformation function
            transforms_test (None, optional): transformation function
            parse_label_func (TYPE, optional): a function to get the label from an input dataframe row
            parse_attention_func (None, optional): a function to get the ground truth label for attention maps
            annotation_file (Optional[str], optional): annotation file
            n_folds (int, optional): the number of folds
            keys_partition (None, optional): it contains n_folds list of strings, each string is a key name
        
        """
        self.mode = mode
        assert self.mode in ['Train', 'Validation', 'Test'], "Wrong mode!"
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test
        self.annotation_file = annotation_file
        self.train_ids = train_ids
        self.val_id = val_id
        self.n_folds = n_folds
        self.keys_partition = keys_partition
        if self.keys_partition is not None:
            assert len(self.keys_partition) == self.n_folds, "len(keys_partition) must be {}".format(self.n_folds)
            # check overlapping
            keys = []
            for k in keys_partition:
                keys.extend(k)
            N = len(keys)
            assert len(np.unique(keys)) == N, "replicate keys in keys_partition"
        self.read_annotaion_file()
        self.parse_label_func = parse_label_func
        self.parse_attention_func = parse_attention_func

    def read_annotaion_file(self):
        data = pickle.load(open(self.annotation_file, 'rb'))

        keys = sorted(list(data.keys()))
        N_keys = len(keys)
        stride = N_keys//self.n_folds

        if self.train_ids is None:
            self.train_ids = list(range(self.n_folds-1))
        else:
            if isinstance(self.train_ids, list):
                assert all([idx< self.n_folds for idx in self.train_ids]), "ids exceeds the number of folds!"

            elif isinstance(self.train_ids, int):
                assert self.train_ids<self.n_folds, "ids exceeds the number of folds!"
                self.train_ids = [self.train_ids]
        if self.keys_partition is None:
            train_keys = []
            for idx in self.train_ids:
                start, end = idx*stride, (idx+1)*stride
                if idx == self.n_folds-1:
                    train_keys.extend(keys[start:])
                else:
                    train_keys.extend(keys[start:end])
        else:
            train_keys = []
            for i in self.train_ids:
                train_keys.extend(self.keys_partition[i])

        if self.val_id is None:
            self.val_id = self.n_folds - 1
        else:
            if isinstance(self.train_ids, list):
                assert self.val_id not in self.train_ids, "data leaking to val set"
            elif isinstance(self.train_ids, int):
                assert self.val_id!=self.train_ids,"val set and train set are the same"
        if self.keys_partition is None:
            if self.val_id==self.n_folds -1:
                val_keys = keys[self.val_id*stride:]
            else:
                val_keys = keys[self.val_id*stride:(self.val_id+1)*stride]
        else:
            val_keys = self.keys_partition[self.val_id]
        train_df = [data[k] for k in train_keys]
        train_df = pd.concat(train_df, ignore_index=True, axis=0)
        val_df = [data[k] for k in val_keys]
        val_df = pd.concat(val_df, ignore_index=True, axis=0)
        if self.mode =='Train':
            self.df = train_df
            self._transform = self.transforms_train
        elif self.mode=='Validation':
            self.df = val_df
            self._transform = self.transforms_test

    @property
    def ids(self):
        return np.arange(len(self.df))
    @property
    def dataset_size(self):
        return len(self.ids)
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self,
        index:int):
        row = self.df.iloc[index]
        label = self.parse_label_func(row)
        img_path = row['path']
        if not os.path.exists(img_path):
            # replace the directory 
            dirname = "/media/Samsung/DISFA"
            if dirname in img_path:
                img_path = img_path.replace(dirname, "../scratch/DISFA")
                if not os.path.exists(img_path):
                    USERDIR = os.environ.get('TACC_USERDIR')
                    img_path = img_path.replace("../scratch/DISFA", os.path.join(USERDIR, 'personal_datasets'))

            dirname = '/media/Samsung/Affectnet_2021March'
            if dirname in img_path:
                img_path = img_path.replace(dirname, "../scratch/Affectnet_2021March")
                if not os.path.exists(img_path):
                    USERDIR = os.environ.get('TACC_USERDIR')
                    img_path = img_path.replace("../scratch/Affectnet_2021March", os.path.join(USERDIR, 'personal_datasets'))

        image = Image.open(img_path).convert("RGB")
        face_ldm = extact_face_landmarks(row)
        if self.parse_attention_func is not None:
            attention_map = self.parse_attention_func(face_ldm) # (N_aus, W, H)
            attention_map = torch.tensor(attention_map).float()
        else:
            attention_map = None
        if self._transform is not None and attention_map is not None:
            image, attention_map = self._transform(image, attention_map)
            return image, label, attention_map
        else:
            image = self._transform(image)
            return image, label

    def _get_all_label(self):
        return self.parse_label_func(self.df)
class Train_Val_Dataset(CV_Dataset):
    def __init__(self, mode: str, 
        transforms_train=None, transforms_test=None, 
        parse_label_func=lambda x:x, parse_attention_func=None, 
        annotation_file:Optional[str]=None):
        #super().__init__()
        """Summary
        
        Args:
            mode (str): the train set, val set ot test set
            transforms_train (None, optional): transformation function
            transforms_test (None, optional): transformation function
            parse_label_func (TYPE, optional): a function to get the label from an input dataframe row
            annotation_file (Optional[str], optional): annotation file
        
        """
        self.mode = mode
        assert self.mode in ['Train', 'Validation', 'Test'], "Wrong mode!"
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test
        self.annotation_file = annotation_file
        self.read_annotaion_file()
        self.parse_label_func = parse_label_func
        self.parse_attention_func = parse_attention_func
    def read_annotaion_file(self):
        data = pickle.load(open(self.annotation_file, 'rb'))
        if self.mode =='Train':
            self.df = data[self.mode]
            self._transform = self.transforms_train
        elif self.mode=='Validation':
            self.df = data[self.mode]
            self._transform = self.transforms_test

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = list(datasets)
        self.datasets_lengths = [len(d) for d in self.datasets]
        max_length = max(self.datasets_lengths)
        # resample the datasets which are less than max_length
        for i in range(len(self.datasets)):
            if len(self.datasets[i])<max_length:
                N = len(self.datasets[i])
                dataset_df = self.datasets[i].df
                for _ in range(max_length//N):
                    self.datasets[i].df = pd.concat([self.datasets[i].df, dataset_df])
                print("dataset {} resampled from {} to {} images".format(i, N, len(self.datasets[i])))
        print("The reference length of datasets is {}".format(max_length))
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets) # it turns out the longer video was sampled less!

    def __len__(self):
        return min([len(d) for d in self.datasets])

class ConcatImageSequenceDataset(ConcatDataset):
    def __init__(self, *datasets):
        self.datasets = list(datasets)
        self.datasets_lengths = [len(d) for d in self.datasets]
        max_length = max(self.datasets_lengths)
        # resample the datasets which are less than max_length
        for i in range(len(self.datasets)):
            if len(self.datasets[i])<max_length:
                N = len(self.datasets[i])
                dataset_sample_seqs = self.datasets[i].sample_seqs
                for _ in range(max_length//N):
                    self.datasets[i].sample_seqs += self.datasets[i].sample_seqs
                print("dataset {} resampled from {} to {} image sequences".format(i, N, len(self.datasets[i])))
        print("The reference length of datasets is {}".format(max_length))