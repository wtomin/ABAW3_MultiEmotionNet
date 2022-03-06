import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from utils.model_utils import get_backbone_from_name
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

# class AU_Attention(nn.Module):
#     def __init__(self, in_channels:int, name: str):
#         super(AU_Attention, self).__init__()
#         self.atten_conv = nn.Conv2d(in_channels, out_channels=1,
#             kernel_size=1)
#         self.batchnorm = nn.BatchNorm2d(1)
#         self.name = name
#     def forward(self, x):
#         x = self.atten_conv(x)
#         x = self.batchnorm(x) # WxW
#         width = x.size(2)
#         bs = x.size(0)
#         x = x.view(bs, width*width)
#         x = F.softmax(x, dim=-1)
#         x = x.view(bs, 1, width, width)
#         return x
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

class AUClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, seq_input):
        bs, seq_len = seq_input.size(0), seq_input.size(1)
        weight = self.fc.weight
        bias = self.fc.bias
        seq_input = seq_input.view((bs*seq_len, 1, -1)) # bs*seq_len, 1, metric_dim
        weight = weight.unsqueeze(0).repeat((bs, 1, 1))  # bs,seq_len, metric_dim
        weight = weight.view((bs*seq_len, -1)).unsqueeze(-1) #bs*seq_len, metric_dim, 1
        inner_product = torch.bmm(seq_input, weight).squeeze(-1).squeeze(-1) # bs*seq_len
        inner_product = inner_product.view((bs, seq_len))
        return inner_product + bias

class Multitask_EmotionNet(pl.LightningModule):
    def __init__(self, backbone:str, tasks: List[str], 
        au_names_list: List[str], emotion_names_list: List[str], va_dim:int,
        AU_metric_dim: int, 
        n_heads:int = 8,
        dropout = 0.3, 
        lr:float=1e-3, 
        T_max:int = 1e4, wd:float=0.,
        AU_cls_loss_func = None,
        EXPR_cls_loss_func = None,
        VA_cls_loss_func = None,
        AU_metric_loss_func = None,
        EXPR_metric_loss_func = None,
        VA_metric_loss_func = None):
        super().__init__()  
        self.tasks = tasks
        self.backbone_CNN = get_backbone_from_name(backbone, pretrained=True, remove_classifier=True)
        self.au_names_list = au_names_list
        self.emotion_names_list = emotion_names_list
        self.va_dim = va_dim
        self.AU_metric_dim = AU_metric_dim
        self.lr = lr
        self.T_max = T_max
        self.wd = wd
        self.AU_cls_loss_func = AU_cls_loss_func
        self.EXPR_cls_loss_func = EXPR_cls_loss_func
        self.VA_cls_loss_func = VA_cls_loss_func

        self.N_emotions = len(self.emotion_names_list)
        if self.AU_cls_loss_func is None:
            self.AU_cls_loss_func = F.binary_cross_entropy_with_logits
        if self.EXPR_cls_loss_func is None:
            self.EXPR_cls_loss_func = F.cross_entropy()
        if self.VA_cls_loss_func is None:
            self.VA_cls_loss_func = F.mse_loss()
        self.AU_metric_loss_func = AU_metric_loss_func
        self.EXPR_metric_loss_func = EXPR_metric_loss_func
        self.VA_metric_loss_func = VA_metric_loss_func
        features_dim = self.backbone_CNN.features_dim
        features_width = self.backbone_CNN.features_width

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
            module = Attention_Metric_Module(features_dim, au_name,
                self.AU_metric_dim)
            self.AU_Metric_Modules.append(module)
        self.AU_Metric_Modules = nn.ModuleList(self.AU_Metric_Modules)
        self.positional_encoding = PositionalEncoding(
            self.AU_metric_dim, dropout = dropout, batch_first = True)
        self.MHA = TransformerEncoderLayerCustom(d_model = self.AU_metric_dim,
            nhead = n_heads, dim_feedforward = 1024,
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
                classifier = nn.Linear(self.AU_metric_dim, len(emotion_names_list))
                #MultiLayerPerceptron(self.AU_metric_dim, hidden_size=[512, 128], out_dim=len(emotion_names_list))
            elif task =='AU':
                classifier = AUClassifier(self.AU_metric_dim, len(au_names_list))
                #MultiLayerPerceptron(len(self.au_names_list)*self.AU_metric_dim, hidden_size=[512, 128], out_dim=len(au_names_list))
            elif task =='VA':
                classifier = nn.Linear(self.AU_metric_dim, va_dim*2)
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
        return outputs, metrics, attention_outputs

    def compute_AU_cls_loss(self, y_hat, y):
        # multilabel classification loss for visible action units
        return self.AU_cls_loss_func(y_hat, y)
    def compute_HuberLoss(self, att_hat, att): # for attention map supervision
        return nn.HuberLoss(delta=0.5)(att_hat, att)
    def parse_multiple_labels(self, y, AU=False, EXPR=False, VA=False):
        if AU and VA and (not EXPR):
            y_au = y[:, :len(self.au_names_list)].float()
            y_va = y[:, len(self.au_names_list):].float()
            return y_au, y_va
        if AU and VA and EXPR:
            y_au = y[:, :len(self.au_names_list)].float()
            y_expr = y[:, len(self.au_names_list)].long()
            y_va = y[:, len(self.au_names_list)+1:].float()
            return y_au,y_expr, y_va
    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        (x_au, y_au), (x_expr, y_expr), (x_va, y_va) = batch['single'] 
        if 'multiple' in batch.keys():
            (x_au_va, y_au_va), (x_au_expr_va, y_au_expr_va) = batch['multiple']
            preds_au_va, metrics_au_va, _ = self(x_au_va)
            preds_au_expr_va, metrics_au_expr_va, _= self(x_au_expr_va)
        total_loss = 0 
        # simple: train with only classification losses, not metric learning losses
        # if metric_loss_func is not None, train with inference losses and metric learning loss
        for task in self.tasks:
            if task =='AU':
                preds, metrics, _ = self(x_au)
                preds_task = preds[task] if 'multiple' not in batch.keys() else torch.cat([preds[task], preds_au_va[task]], dim=0)
                labels_task = y_au if 'multiple' not in batch.keys() else torch.cat([y_au, self.parse_multiple_labels(y_au_va, AU=True, VA=True)[0]], dim=0)
                metrics_task = metrics[task] if 'multiple' not in batch.keys() else torch.cat([metrics[task], metrics_au_va[task]], dim=0)
                loss = self.AU_cls_loss_func(preds_task, labels_task)
                if self.AU_metric_loss_func is not None:
                    metric_loss = 0
                    for i_au in range(len(self.au_names_list)):
                        metric_loss += (1/len(self.au_names_list))*self.AU_metric_loss_func(metrics_task[:, i_au,:], labels_task[:, i_au])
                    loss += metric_loss
                
            elif task =='EXPR':
                preds, metrics, _ = self(x_expr)
                preds_task = preds[task] if 'multiple' not in batch.keys() else torch.cat([preds[task], preds_au_expr_va[task]], dim=0)
                labels_task = y_expr if 'multiple' not in batch.keys() else torch.cat([y_expr, self.parse_multiple_labels(y_au_expr_va, AU=True, EXPR=True, VA=True)[1]], dim=0)
                metrics_task = metrics[task] if 'multiple' not in batch.keys() else torch.cat([metrics[task], metrics_au_expr_va[task]], dim=0)
                loss = self.EXPR_cls_loss_func(preds_task, labels_task)
                if self.EXPR_metric_loss_func is not None:
                    metric_loss = self.EXPR_metric_loss_func(metrics_task, labels_task)
                    loss += metric_loss
            elif task =='VA':
                preds, metrics, _ = self(x_va)
                preds_task = preds[task] if 'multiple' not in batch.keys() else torch.cat([preds[task], preds_au_va[task], preds_au_expr_va[task]], dim=0)
                labels_task = y_va if 'multiple' not in batch.keys() else torch.cat([y_va, 
                self.parse_multiple_labels(y_au_va, AU=True, VA=True)[-1] ,
                self.parse_multiple_labels(y_au_expr_va, AU=True, EXPR=True, VA=True)[-1]], dim=0)
                metrics_task = metrics[task] if 'multiple' not in batch.keys() else torch.cat([metrics[task], metrics_au_va[task], metrics_au_expr_va[task]], dim=0)
                loss = self.VA_cls_loss_func(preds_task, labels_task)
                if self.VA_metric_loss_func is not None:
                    metric_loss = self.VA_metric_loss_func(metrics_task, labels_tasks)
                    loss+=metric_loss
            self.log('loss_{}'.format(task), loss, on_step=True, on_epoch=True, 
                prog_bar=True, logger=True)
            total_loss += loss
        self.log('total_loss'.format(task), total_loss, on_step=True, on_epoch=True, 
            prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # multiple validation datasets
        # import pdb; pdb.set_trace()
        x, y = batch 
        preds,_,_ = self(x)
        return torch.cat([preds['AU'], preds['EXPR'], preds['VA']], dim=-1), y

    def compute_F1_numpy(self,y_hat, y):
        return f1_score(y_hat.numpy(), y.numpy())
    def compute_L1_distance(self, y_hat, y):
        return F.l1_loss(y_hat, y)
    
    def save_val_metrics_to_logging(self, task, preds_task, labels_task, save_name):
        metric_values, metric_task = get_metric_func(task)(preds_task.numpy(), labels_task.numpy())
        if task != 'VA':
            self.log('{}_F1'.format(save_name), metric_values[0], on_epoch=True, logger=True)
            self.log('{}_Acc'.format(save_name), metric_values[0], on_epoch=True, logger=True)
        else:
            self.log('{}_{}'.format(save_name, 'valence'), metric_values[0], on_epoch=True, logger=True)  
            self.log('{}_{}'.format(save_name, 'arousal'), metric_values[1], on_epoch=True, logger=True) 
        return metric_task
    def validation_single_task(self, dataloader_idx, preds, labels):
        if len(labels.size())==1:
            # expr
            preds_task = F.softmax(preds['EXPR'], dim=-1).argmax(-1).int()
            labels_task = labels.int()
            metric_8_task = self.save_val_metrics_to_logging('EXPR', preds_task, 
                labels_task, save_name="D{}/EXPR8".format(dataloader_idx))
            mask = labels_task<7
            if sum(mask)>0:
                metric_7_task = self.save_val_metrics_to_logging('EXPR', preds_task[mask], 
                    labels_task[mask], save_name="D{}/EXPR7".format(dataloader_idx))
            return metric_8_task

        elif labels.size(1)==12:
            # au
            preds_task = (torch.sigmoid(preds['AU'])>0.5).int()
            labels_task = labels.int()
            metric_aus = self.save_val_metrics_to_logging('AU', preds_task, 
                labels_task, save_name="D{}/AU".format(dataloader_idx))
            return metric_aus
        elif labels.size(1) == 2:
            # va
            preds_task = preds['VA'].float()
            labels_task = labels
            metric_va = self.save_val_metrics_to_logging('VA', preds_task, 
                labels_task, save_name="D{}/VA".format(dataloader_idx))
            return metric_va
    def turn_mul_emotion_preds_to_dict(self, preds):
        # au, expr , va
        return {'AU': preds[..., :len(self.au_names_list)], 
        'EXPR': preds[..., len(self.au_names_list): len(self.au_names_list)+len(self.emotion_names_list)],
        'VA': preds[..., -2:]}
    def validation_on_single_dataloader(self, dataloader_idx, val_dl_outputs):
        num_batches = len(val_dl_outputs)
        preds = torch.cat([x[0] for x in val_dl_outputs], dim=0).cpu()
        labels= torch.cat([x[1] for x in val_dl_outputs], dim=0).cpu()
        preds = self.turn_mul_emotion_preds_to_dict(preds)
        if len(labels.size())>1 and labels.size(1) == len(self.au_names_list)+2:
            # au_va
            metrics_aus = self.validation_single_task(dataloader_idx, preds,
             labels[:, :len(self.au_names_list)])
            metrics_va = self.validation_single_task(dataloader_idx, preds,
             labels[:, len(self.au_names_list):])
            return metrics_aus+metrics_va

        elif len(labels.size())>1 and labels.size(1) == len(self.au_names_list)+1+2:
            # au_expr_va
            metrics_aus = self.validation_single_task(dataloader_idx, preds,
             labels[:, : len(self.au_names_list)])
            metrics_expr = self.validation_single_task(dataloader_idx, preds,
             labels[:, len(self.au_names_list)])
            metrics_va = self.validation_single_task(dataloader_idx, preds,
             labels[:, -2:])
            return metrics_aus+metrics_expr+metrics_va
        else:
            return self.validation_single_task(dataloader_idx, preds, labels)

    def validation_epoch_end(self, validation_step_outputs):
        #check the validation step outputs
        # import pdb; pdb.set_trace()
        num_dataloaders = len(validation_step_outputs) # three or five
        total_metric = 0
        for dataloader_idx in range(num_dataloaders):
            val_dl_outputs = validation_step_outputs[dataloader_idx]
            idx_metric = self.validation_on_single_dataloader(dataloader_idx, val_dl_outputs)
        self.log('val_total', total_metric, on_epoch=True, logger=True) 

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

 













