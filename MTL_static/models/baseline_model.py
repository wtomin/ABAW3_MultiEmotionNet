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

class Baseline_Model(InceptionV3MTModel):
    def __init__(*args, **kwargs):
        InceptionV3MTModel.__init__(*args, **kwargs)
        
    def configure_architecture(self):
        emotion_metric_modules = []
        for task in self.tasks:
            if task =='EXPR':
                module = nn.Sequential(nn.Linear(self.features_dim, self.AU_metric_dim),
                    nn.ReLU())
            elif task =='AU':
                module = nn.ModuleList([nn.Sequential(nn.Linear(self.features_dim, self.AU_metric_dim),
                    nn.ReLU()) for _ in range(len(self.au_names_list))])
            elif task =='VA':
                module = nn.Sequential(nn.Linear(self.features_dim, self.AU_metric_dim),
                    nn.ReLU())
            emotion_metric_modules.append(module)
        self.emotion_metric_modules = nn.ModuleList(emotion_metric_modules)
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
        feature_maps = F.max_pool2d(feature_maps, kernel_size=17).squeeze(-1).squeeze(-1)

        metrics = {}
        AU_metrics = []
        au_names = []
        outputs = {}
        i_classifier = self.tasks.index('AU')
        for i_au, au_module in enumerate(self.emotion_metric_modules[i_classifier]):
            AU_metrics.append(au_module(feature_maps))
        AU_metrics = torch.stack(AU_metrics, dim=1)  # stack the metrics as sequence
        AU_metrics_with_labels = AU_metrics
        outputs['AU'] = self.emotion_classifiers[i_classifier](AU_metrics_with_labels)
        metrics['AU'] = AU_metrics_with_labels

        i_classifier = self.tasks.index('EXPR')
        EXPR_metrics = self.emotion_metric_modules[i_classifier](feature_maps)

        # EXPR classifier
        outputs['EXPR'] = self.emotion_classifiers[i_classifier](EXPR_metrics)
        metrics['EXPR'] = EXPR_metrics

        # VA classifier
        i_classifier = self.tasks.index('VA')
        VA_metrics = self.emotion_metric_modules[i_classifier](feature_maps)
        outputs['VA'] = self.emotion_classifiers[i_classifier](VA_metrics)
        metrics['VA'] = VA_metrics
        return outputs, metrics
















