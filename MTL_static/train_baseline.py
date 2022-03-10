import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.baseline_model import Baseline_Model
from data.ABAW3_DataModule import get_MTL_datamodule
from utils.data_utils import train_transforms, test_transforms, Tversky_Loss_with_Logits, FocalTversky_Loss_with_Logits, CCCLoss
from PATH import PATH
PRESET_VARS = PATH()
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning import distances
from utils.data_utils import get_ap_and_an, get_quadruplet

torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type(torch.FloatTensor)
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--ckp-save-dir', type=str, default='Baseline_Model')
    parser.add_argument('--exp-name', type=str, default='experiment_1')
    parser.add_argument('--find_best_lr', action="store_true")
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--resume_ckp', type=str, default=None,
        help='if set resume ckp, load it before training')

    args = parser.parse_args()
    au_names_list = PRESET_VARS.ABAW3.categories['AU']
    emotion_names_list = PRESET_VARS.ABAW3.categories['EXPR']
    pos_weight = [7.1008924, 15.63964869, 5.47108051, 2.80360066, 1.5152332, 1.89083564, 3.04637044, 34.04600245, 32.47861156, 36.76637801, 0.58118674, 11.1586486]
    pos_weight = torch.tensor(pos_weight)
    pos_weight = pos_weight.float().cuda()
    def classification_loss_func(y_hat, y):
        loss1 = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight= pos_weight)
        return loss1
    def regression_loss_func(y_hat, y):
        loss1 =  CCCLoss(digitize_num=1)(y_hat[:, 0], y[:, 0]) + CCCLoss(digitize_num=1)(y_hat[:, 1], y[:, 1])
        return loss1
    class_weights = [1.05169857, 11.99656373, 17.33464893, 15.57124886, 2.14378498, 2.59458996, 6.41378336, 1.] 
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights.float().cuda()
    def cross_entropy_loss(y_hat, y):
        Num_classes = y_hat.size(-1)
        return F.cross_entropy(y_hat, y, weight=class_weights[:Num_classes])

    def AU_metric_loss(AU_embedding, y):
        """
        Use tripletmargin loss for the cosine distance between positive and negative pairs
        """
        loss = losses.TripletMarginLoss(margin=0.2, smooth_loss=True, distance=distances.CosineSimilarity())
        return loss(AU_embedding, y)
    def EXPR_metric_loss(EXPR_embedding, y, margin=0.15):
        mask = y>0 # apply triplet loss except for neutral 
        Q_loss =  quadruplet_loss(EXPR_embedding[mask], y[mask], margin=margin)
        return Q_loss

    def quadruplet_loss(embedding, y, margin = 0, multiplier = 2):
        # quadruplet has reduced the number of tuples by over 10.
        a_idx, p_idx, n_adj_idx, n_non_idx = get_quadruplet(y)
        if len(a_idx)>0:
            loss_apsn = losses.TripletMarginLoss(margin=margin, smooth_loss=True, distance=distances.CosineSimilarity())
            loss1 = loss_apsn(embedding, y, (a_idx, p_idx, n_adj_idx))
            loss_apan = losses.TripletMarginLoss(margin=multiplier*margin, smooth_loss=True, distance=distances.CosineSimilarity())
            loss2 = loss_apan(embedding, y, (a_idx, p_idx, n_non_idx))
            return loss1+loss2
        else:
            return 0

    def VA_metric_loss(VA_embedding, y_va, margin=0.08):
        # assumption: same class will have similar VA embedding in euclidean space
        loss = losses.TripletMarginLoss(margin=margin, smooth_loss=True) # use Euclidean space by default
        indices_tuple = get_ap_and_an(y_va, delta_ap=0.2, delta_an=0.6) # va labels used for triplet indices 
        return loss(VA_embedding, y_va[:, 0], indices_tuple) # here use dummy 2D labels, it won't contribute to the final loss

    model = Baseline_Model('inception_v3', ['AU', 'EXPR', 'VA'], 
        au_names_list, emotion_names_list, va_dim=1, AU_metric_dim = 16,
        lr = args.lr, AU_cls_loss_func = classification_loss_func,
        EXPR_cls_loss_func = cross_entropy_loss,
        VA_cls_loss_func = regression_loss_func,
        AU_metric_loss_func = None, #AU_metric_loss,
        EXPR_metric_loss_func = None, #EXPR_metric_loss,
        VA_metric_loss_func = None, #VA_metric_loss,
        wd=0)

    img_size = 299

    dm = get_MTL_datamodule(video=False, img_size = img_size,
        batch_size=24, num_workers_train=8, num_workers_test=8)

    ckp_dir = os.path.join(args.ckp_save_dir, args.exp_name)
    # on TACC platform, save the model in USERDIR
    if os.environ.get('TACC_USERDIR') is not None:
        USERDIR =  os.environ.get('TACC_USERDIR')
        ckp_dir = os.path.join(USERDIR, ckp_dir)
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckp_callback1 = ModelCheckpoint(monitor='val_total', mode='max',
        dirpath = ckp_dir,
        filename = 'model-{epoch:02d}-{val_total:.2f}',
        save_top_k = 1,
        save_last= True)

    tb_logger = pl_loggers.TensorBoardLogger(ckp_dir)
    trainer = Trainer(gpus=1, benchmark=True,
        default_root_dir = ckp_dir, logger = tb_logger, log_every_n_steps=100, 
        max_steps = 3e5,
        callbacks =[lr_monitor, ckp_callback1],
        resume_from_checkpoint=args.resume_ckp)
        # limit_train_batches = 0.01, 
        # limit_val_batches = 0.01)

    if args.find_best_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule = dm, 
            min_lr = 1e-5, max_lr = 1e-1)
        fig = lr_finder.plot(suggest=True)
        fig.show()
    else:
        trainer.fit(model, datamodule=dm)