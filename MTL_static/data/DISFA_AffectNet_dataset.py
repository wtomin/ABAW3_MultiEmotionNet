import numpy as np
import torch
from PATH import PATH
PRESET_VARS = PATH()
from typing import Optional, List, Union
#import sys
#sys.path.append('..')
from data.dataset import CV_Dataset, Train_Val_Dataset,ConcatDataset, DataModuleBase


def parse_disfa_label(df_row):
    aus = PRESET_VARS.DISFA.categories['AU']
    return (df_row[aus].values>=2).astype(np.int32)
def parse_affectnet_label(df_row):
    expr = df_row['expression']
    assert expr<8, "assume expr has 8 categories"
    v, a = df_row['valence'], df_row['arousal']
    return np.array([expr, v, a])

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

def get_affectnet_train_valset(transforms_train=None, transforms_test=None, 
        parse_label_func=lambda x:x,  parse_attention_func=None,
        annotation_file:Optional[str]=None):
    trainset = Train_Val_Dataset('Train', transforms_train, transforms_test,
        parse_label_func, parse_attention_func, annotation_file)
    valset = Train_Val_Dataset('Validation', transforms_train, transforms_test,
        parse_label_func, parse_attention_func,annotation_file)
    return trainset, valset
class IterativeCombinedDataLoader:
    def __init__(self, train_loaders):
        self.train_loaders = train_loaders
        self.inner_iterators = [iter(da) for da in self.train_loaders]
        self.max = min([len(da) for da in self.train_loaders])
        self.n = 0
    def __iter__(self,):
        self.__init__(self.train_loaders)
        return self
    def __next__(self,):
        if self.n < self.max:
            xs = [next(x) for x in self.inner_iterators]
            self.n=self.n+1
            return xs
        raise StopIteration()

class DISFA_AffectNet_DataModule(DataModuleBase):
    def __init__(self, trainsets, valsets, batch_size, sampler, batch_sampler, 
    	*args, **kwargs):
        super(DISFA_AffectNet_DataModule, self).__init__(*args, **kwargs)
        self.trainsets = trainsets # leave the training set as list
        if isinstance(self.trainsets , list):
            if len(self.trainsets) == 1:
                self.trainsets = self.trainsets[0]
            else:
                print("Train sets are {}".format(len(self.trainsets)))
                for i, dataset in enumerate(self.trainsets):
                    print('train set {}: {} samples'.format(i, len(dataset)))
                self.trainsets = ConcatDataset(*self.trainsets)
                print('After ConcatDataset, samples:{}'.format(len(self.trainsets)))
        self.valsets = valsets
        self.batch_size = batch_size
        self.sampler = sampler # only apply to train dataloader
        self.batch_sampler = batch_sampler# only apply to train dataloader
        print("sampler and batch_sampler are only applied to train datasets.")

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            print('loaded')

    def train_dataloader(self):
        if self.batch_sampler is None:
            return torch.utils.data.DataLoader(self.trainsets,
                batch_size =  self.batch_size,
                num_workers = self.num_workers_train,
                shuffle = True,
                drop_last = True)
        else:
            return torch.utils.data.DataLoader(self.trainsets,
                   batch_sampler = self.batch_sampler,
                num_workers = self.num_workers_train)

    def val_dataloader(self):
        if not isinstance(self.valsets, list):
            return torch.utils.data.DataLoader(self.valsets,
                batch_size = self.batch_size, num_workers = self.num_workers_test,
                shuffle=False, drop_last = False)
        else:
            return [torch.utils.data.DataLoader(valset,
                batch_size = self.batch_size, num_workers = self.num_workers_test,
                shuffle=False, drop_last = False) for valset in self.valsets]

def get_disfa_affectnet_datamoudle(train_ids, val_id, transforms_train, transforms_test, batch_size,
    sampler=None, batch_sampler=None, n_folds=3,keys_partition =None,
    num_workers_train:int = 0, num_workers_test:int = 0):

    annot_file_disfa = PRESET_VARS.DISFA.data_file
    annot_file_affectnet = PRESET_VARS.AffectNet.data_file
    trainset_disfa, valset_disfa = get_disfa_train_valset(train_ids, val_id, transforms_train, transforms_test,
        parse_disfa_label, None, annot_file_disfa, n_folds, keys_partition)
    trainset_aff, valset_aff = get_affectnet_train_valset(transforms_train, transforms_test,
        parse_affectnet_label, None, annot_file_affectnet)
    datamodule = DISFA_AffectNet_DataModule([trainset_disfa, trainset_aff], 
        [valset_disfa, valset_aff], batch_size, sampler, batch_sampler,
    	num_workers_train, num_workers_test)
    return datamodule







        


