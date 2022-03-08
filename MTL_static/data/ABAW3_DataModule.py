import numpy as np
import torch
from PATH import PATH
PRESET_VARS = PATH()
from typing import Optional, List, Union
from data.dataset import ImageDataset, ImageSequenceDataset, ConcatDataset, ConcatImageSequenceDataset, DataModuleBase
from utils.sampler import RandomSubsetShuffledSampler
from utils.data_utils import train_transforms, test_transforms
def parse_emotion_label(df_row):
    AU_names = PRESET_VARS.ABAW3.categories['AU']
    labels = []
    if AU_names[0] in df_row.keys():
        labels.append(df_row[AU_names].values.astype(np.float32))
    if 'label' in df_row.keys():
        labels.append(df_row['label'])
    if 'valence' in df_row.keys():
        labels.append(df_row[['valence', 'arousal']].values.astype(np.float32))
    if len(labels)==1:
        return labels[0]
    else:
        for i in range(len(labels)):
            if not isinstance(labels[i], np.ndarray):
                labels[i] = np.array(labels[i]).reshape((-1,))
        return np.concatenate(labels, axis=-1) # concatenate all emotion labels at the last axis

def get_Dataset_TrainVal(annotation_file:str, video = False, transforms_train=None, transforms_test=None, 
    parse_label_func=parse_emotion_label,
    seq_len = None):
    if video:
        assert seq_len is not None, "seq_len needs specification"
        Dataset = ImageSequenceDataset
        trainset = Dataset('Train', seq_len, transforms_train, transforms_test, parse_label_func, annotation_file)
        valset = Dataset('Validation', seq_len, transforms_train, transforms_test, parse_label_func, annotation_file)
    else:
        Dataset = ImageDataset
        trainset = Dataset('Train', transforms_train, transforms_test, parse_label_func, annotation_file)
        valset = Dataset('Validation', transforms_train, transforms_test, parse_label_func, annotation_file)
    return trainset, valset

def concatenate_datasets(trainsets, video=False):
    if not video:
        return ConcatDataset(*trainsets)
    else:
        return ConcatImageSequenceDataset(*trainsets)

class MTL_DataModule(DataModuleBase):
    def __init__(self, video: bool, trainsets1, valsets1, batch_size1, trainsets2=None, valsets2=None, batch_size2=None, 
        *args, **kwargs):
        super(MTL_DataModule, self).__init__(*args, **kwargs)
        self.trainsets1 = concatenate_datasets(trainsets1, video)
        self.valsets1 = valsets1
        self.batch_size1 = batch_size1
        if trainsets2 is not None:
            self.trainsets2 = concatenate_datasets(trainsets2, video)
        else:
            self.trainsets2 = None
        self.valsets2 = valsets2
        self.batch_size2 = batch_size2

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            print('loaded')
    def train_dataloader(self):
        D1 = torch.utils.data.DataLoader(self.trainsets1,
                batch_size =  self.batch_size1,
                num_workers = self.num_workers_train,
                sampler = RandomSubsetShuffledSampler(list(range(len(self.trainsets1))), splits = 8),
                drop_last = True)
        if self.trainsets2 is not None:
            D2 = torch.utils.data.DataLoader(self.trainsets2,
                    batch_size =  self.batch_size2,
                    num_workers = self.num_workers_train,
                    sampler = RandomSubsetShuffledSampler(list(range(len(self.trainsets2))), splits = 4),
                    drop_last = True)
            return {"single": D1, "multiple": D2}
        else:
            return {'single': D1}
    def val_dataloader(self):
        val_batch_size = self.batch_size1*3
        val_loaders = [torch.utils.data.DataLoader(valset,
                batch_size = val_batch_size, num_workers = self.num_workers_test,
                shuffle=False, drop_last = False) for valset in self.valsets1]
        if self.valsets2 is not None:
            val_loaders+= [torch.utils.data.DataLoader(valset,
                    batch_size = val_batch_size, num_workers = self.num_workers_test,
                    shuffle=False, drop_last = False) for valset in self.valsets2]
        return val_loaders # five or three val loaders

def get_MTL_datamodule(video, img_size, batch_size, seq_len=None, num_workers_train=0, num_workers_test=0):
    dataset1 = get_Dataset_TrainVal("create_annotation_file/AU/AU_annotations.pkl", video=video,
        transforms_train=train_transforms(img_size), transforms_test=test_transforms(img_size), 
        seq_len=seq_len)
    dataset2 = get_Dataset_TrainVal("create_annotation_file/EXPR/EXPR_annotations.pkl", video=video,
        transforms_train=train_transforms(img_size), transforms_test=test_transforms(img_size), 
        seq_len=seq_len)
    dataset3 = get_Dataset_TrainVal("create_annotation_file/VA/VA_annotations.pkl", video=video,
        transforms_train=train_transforms(img_size), transforms_test=test_transforms(img_size), 
        seq_len=seq_len)

    dataset4 = get_Dataset_TrainVal("create_annotation_file/MTL/AU_VA_annotations.pkl", video=video,
        transforms_train=train_transforms(img_size), transforms_test=test_transforms(img_size), 
        seq_len=seq_len)
    dataset5 = get_Dataset_TrainVal("create_annotation_file/MTL/AU_EXPR_VA_annotations.pkl", video=video,
        transforms_train=train_transforms(img_size), transforms_test=test_transforms(img_size), 
        seq_len=seq_len)
    dm = MTL_DataModule(video,
        [dataset1[0], dataset2[0], dataset3[0]], 
        [dataset1[1], dataset2[1], dataset3[1]], batch_size,
        # None, None, None,
        [dataset4[0], dataset5[0]], [dataset4[1], dataset5[1]],
        max(1, int(batch_size*(1/15))),
        num_workers_train = num_workers_train, 
        num_workers_test=num_workers_test)

    return dm