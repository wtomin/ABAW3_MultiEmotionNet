import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from models.model import InceptionV3MTModel, AUClassifier
from utils.data_utils import  get_metric_func
#from miners.triplet_margin_miner import TripletMarginMiner
from pytorch_metric_learning import losses, miners, distances, reducers
import torchmetrics
from sklearn.metrics import f1_score
import numpy as np
from torch.nn import TransformerEncoderLayer
from typing import Optional, Any
from torch import Tensor
import math
import geotorch
from PATH import PATH
PRESET_VARS = PATH()
#from utils.data_utils import compute_center_contrastive_loss, get_center_delta

class PositionalEncoding(nn.Module):
    def __init__(self,
        emb_size: int,
        dropout = float,
        maxlen: int = 5000,
        batch_first = False):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000)/emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) #(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) #(maxlen, 1, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.batch_first = batch_first

    def forward(self, token_embedding:Tensor):
        if self.batch_first:
            return self.dropout(token_embedding +
                self.pos_embedding.transpose(0,1)[:,:token_embedding.size(1)])
        else:
            return self.dropout(token_embedding +
                self.pos_embedding[:token_embedding.size(0), :])


class TransformerEncoderLayerCustom(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attention_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention_weights

class Attention_Metric_Module(nn.Module):
    def __init__(self, in_channels: int, name:str, 
        metric_dim:int):
        super(Attention_Metric_Module, self).__init__()
        
        self.name = name
        self.in_channels = in_channels
        self.metric_dim =  metric_dim
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=self.metric_dim, 
            kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.metric_dim),
            nn.ReLU())
        self.Conv2 = nn.Sequential(
            nn.Conv2d(self.metric_dim, out_channels =self.metric_dim, 
                kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm2d(self.metric_dim),
            nn.ReLU())
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.metric_dim, self.metric_dim, bias=True)

    def forward(self, x, attention):
        x = x*attention + x
        x = self.Conv1(x) # bsxn_outxWxW
        x = self.Conv2(x) # bs x n_out x 1 x 1
        x = self.GAP(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

class Multitask_EmotionNet(InceptionV3MTModel):
    def __init__(*args, **kwargs):
        InceptionV3MTModel.__init__(*args, **kwargs)
    def configure_architecture(self):
        # attention branches
        self.AU_attention_convs = nn.Sequential(
            nn.Conv2d(768, out_channels=(len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs))*4,
            kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d((len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs))*4),
            nn.ReLU(),
            nn.Conv2d((len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs))*4, out_channels = len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs),
                kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs)),
            nn.ReLU())
        self.AU_attention_map_module = nn.Sequential(
            nn.Conv2d(len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs), out_channels=len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs),
                kernel_size = 1, stride=1, padding=0, bias=True),
            nn.Sigmoid())
        self.AU_attention_classification_module=nn.Sequential(
            nn.Conv2d(len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs), out_channels=len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs),
                kernel_size = 1, stride=1, padding=0, bias=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten())
        self.AU_Metric_Modules = []
        for au_name in self.au_names_list + PRESET_VARS.Hidden_AUs:
            module = Attention_Metric_Module(self.features_dim, au_name,
                self.AU_metric_dim)
            self.AU_Metric_Modules.append(module)
        self.AU_Metric_Modules = nn.ModuleList(self.AU_Metric_Modules)
        self.positional_encoding = PositionalEncoding(
            self.AU_metric_dim, dropout = self.dropout, batch_first = True)
        self.MHA = TransformerEncoderLayerCustom(d_model = self.AU_metric_dim,
            nhead = self.n_heads, dim_feedforward = 1024,
            activation='gelu', batch_first=True)

        # Learn the orthogonal matrices to transform AU embeddings to EXPR-VA feature space.
        self.rotation_matrices = []

        for i_au in range(len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs)):
            matrix = nn.Linear(self.AU_metric_dim, self.AU_metric_dim, bias=False)
            geotorch.orthogonal(matrix, 'weight')
            self.rotation_matrices.append(matrix)
        self.rotation_matrices = nn.ModuleList(self.rotation_matrices)

        # change the classifier to multitask emotion classifiers with K splits
        emotion_classifiers = []
        for task in self.tasks:
            if task =='EXPR':
                classifier = nn.Linear(self.AU_metric_dim, len(self.emotion_names_list))
                #MultiLayerPerceptron(self.AU_metric_dim, hidden_size=[512, 128], out_dim=len(emotion_names_list))
            elif task =='AU':
                classifier = AUClassifier(self.AU_metric_dim, len(self.au_names_list))
                #MultiLayerPerceptron(len(self.au_names_list)*self.AU_metric_dim, hidden_size=[512, 128], out_dim=len(au_names_list))
            elif task =='VA':
                classifier = nn.Linear(self.AU_metric_dim, self.va_dim*2)
                # MultiLayerPerceptron(self.AU_metric_dim, hidden_size=[512, 128], out_dim=va_dim*2)
            emotion_classifiers.append(classifier)

        self.emotion_classifiers = nn.ModuleList(emotion_classifiers)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        feature_maps = self.backbone_CNN(x)
        x1 = self.AU_attention_convs(feature_maps)
        atten_maps = self.AU_attention_map_module(x1)
        atten_preds = self.AU_attention_classification_module(x1)
        attention_outputs = [atten_maps, atten_preds]

        metrics = {}
        AU_metrics = []
        au_names = []
        outputs = {}
        for i_au, au_module in enumerate(self.AU_Metric_Modules):
            atten_map = atten_maps[:, i_au, ...].unsqueeze(1)
            au_metric = au_module(feature_maps, atten_map)
            au_name = au_module.name
            AU_metrics.append(au_metric)
            au_names.append(au_name)
        AU_metrics = torch.stack(AU_metrics, dim=1)  # stack the metrics as sequence
        input_seq = self.positional_encoding(AU_metrics)
        output_seq = self.MHA(input_seq)[0]
        AU_metrics_with_labels = output_seq[:, :len(self.au_names_list), :] # taken only the embeddings with AU labels
        i_classifier = self.tasks.index('AU')
        outputs['AU'] = self.emotion_classifiers[i_classifier](AU_metrics_with_labels)
        metrics['AU'] = AU_metrics_with_labels
        AU_metrics = output_seq

        # rotate the AU embeddings by learned rotation matrices
        EXPR_VA_metrics = []
        for i_au in range(len(self.au_names_list) + len(PRESET_VARS.Hidden_AUs)):
            au_metric = AU_metrics[:, i_au, ...]
            rotated = self.rotation_matrices[i_au](au_metric)
            EXPR_VA_metrics.append(rotated)
        EXPR_VA_metrics = torch.stack(EXPR_VA_metrics, dim=1).mean(1)

        # EXPR classifier
        i_classifier = self.tasks.index('EXPR')
        outputs['EXPR'] = self.emotion_classifiers[i_classifier](EXPR_VA_metrics)
        metrics['EXPR'] = EXPR_VA_metrics

        # VA classifier
        i_classifier = self.tasks.index('VA')
        outputs['VA'] = self.emotion_classifiers[i_classifier](EXPR_VA_metrics)
        metrics['VA'] = EXPR_VA_metrics
        return outputs, metrics

    def configure_optimizers(self,):
        paramters_dict = [ {'params': [param for name, param in self.named_parameters() if 'rotation_matrices' not in name]}, 
        {'params':self.rotation_matrices.parameters() , 'lr':self.lr*10}]
        optimizer = torch.optim.SGD(paramters_dict, momentum=0.9,
                lr=self.lr,
                weight_decay = self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.T_max)

        return {
        'optimizer': optimizer,
        'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

 













