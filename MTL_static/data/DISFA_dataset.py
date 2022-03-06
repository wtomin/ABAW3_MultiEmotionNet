import numpy as np
import torch
from PATH import PATH
PRESET_VARS = PATH()
from typing import Optional, List, Union

from data.dataset import CV_Dataset, DataModuleBase
from utils.data_utils import landmarks_to_attention_map

def parse_disfa_label(df_row):
    aus = PRESET_VARS.DISFA.categories['AU']
    return df_row[aus].values.astype(np.int32)
def parse_disfa_attentions(face_shape, in_size=(224, 224), out_size = (112, 112)):
    aus = PRESET_VARS.DISFA.categories['AU']
    attention_maps = []
    for au_name in aus:
        attention_map = landmarks_to_attention_map(au_name, face_shape, in_size=in_size, out_size=out_size)
        attention_maps.append(attention_map)
    return np.stack(attention_maps, axis=0)

def get_disfa_train_valset(train_ids:Union[int, List[int]]=None, val_id:Optional[int]=None, 
        transforms_train=None, transforms_test=None, parse_label_func=lambda x:x,  parse_attention_func=None,
        annotation_file:Optional[str]=None, 
        n_folds:int=3,
        keys_partition = None):
    trainset = CV_Dataset('Train', train_ids, val_id, transforms_train, transforms_test,
        parse_label_func, parse_attention_func, annotation_file, n_folds, keys_partition)
    valset = CV_Dataset('Validation', train_ids, val_id, transforms_train, transforms_test,
        parse_label_func, parse_attention_func,annotation_file, n_folds, keys_partition)

    return trainset, valset

class DISFA_DataModule(DataModuleBase):
    def __init__(self, trainset, valset, batch_size, sampler, batch_sampler, 
    	*args, **kwargs):
        super(DISFA_DataModule, self).__init__(*args, **kwargs)
        self.trainset = trainset
        self.valset = valset
        self.batch_size = batch_size
        self.sampler = sampler # only apply to train dataloader
        self.batch_sampler = batch_sampler# only apply to train dataloader
        print("sampler and batch_sampler are only applied to train datasets.")

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            print("Train dataset loaded from DISFA dataset. Parts :{}, number of images: {}".format(self.trainset.train_ids, len(self.trainset)))
            print("Validation dataset loaded from DISFA dataset. Parts :{}, number of images: {}".format(self.valset.val_id, len(self.valset)))
    def train_dataloader(self):
        if self.batch_sampler is None:
            return torch.utils.data.DataLoader(self.trainset,
                   sampler = self.sampler,
                batch_size = self.batch_size,
                num_workers = self.num_workers_train,
                shuffle = True,
                drop_last = True)
        else:
            return torch.utils.data.DataLoader(self.trainset,
                   batch_sampler = self.batch_sampler,
                num_workers = self.num_workers_train)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valset,
            batch_size = self.batch_size, num_workers = self.num_workers_test,
            shuffle=False, drop_last = False)

def get_disfa_datamoudle(train_ids, val_id, transforms_train, transforms_test, batch_size,
    sampler=None, batch_sampler=None, n_folds=3, keys_partition =None,
    num_workers_train:int = 0, num_workers_test:int = 0):

    annot_file = PRESET_VARS.DISFA.data_file
    trainset, valset = get_disfa_train_valset(train_ids, val_id, transforms_train, transforms_test,
        parse_disfa_label, parse_disfa_attentions, annot_file, n_folds, keys_partition)
    datamodule = DISFA_DataModule(trainset, valset, batch_size, sampler, batch_sampler,
    	num_workers_train, num_workers_test)
    return datamodule







        


