import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.static_linear_model import Multitask_EmotionNet
from data.dataset import ImageSequenceDataset, ImageDataset
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
from utils.data_utils import AU_metric, EXPR_metric, VA_metric
torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type(torch.FloatTensor)
import pickle
from tqdm import tqdm
import numpy as np
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

def return_metric(preds, labels, task):
    if task == 'AU':
        # _, res = AU_metric(preds, labels)
        res =  AU_metric(preds, labels)[0][0]
    elif task == 'EXPR':
        # _, res = EXPR_metric(preds, labels)
        res = EXPR_metric(preds, labels)[0][0]
    else:
        # _, res = VA_metric(preds, labels)
        res = EXPR_metric(preds, labels)[0]
        res = np.mean(res)
    return res
def temporal_smooth_function(feature_sequence, alpha):
    after_smooth = [feature_sequence[0]]
    for i in range(1, len(feature_sequence)):
        res = (after_smooth[i-1]*alpha + feature_sequence[i])/(1+alpha)
        after_smooth.append(res)
    return torch.stack(after_smooth, dim=0)
def get_Dataset_Val(annotation_file,  transforms_train=None, transforms_test=None, 
    parse_label_func=parse_emotion_label,
    seq_len = None):
    video = False
    if video:
        assert seq_len is not None, "seq_len needs specification"
        Dataset = ImageSequenceDataset
        testset = Dataset('Validation', seq_len, transforms_train, transforms_test, parse_label_func, annotation_file)
    else:
        Dataset = ImageDataset
        testset = Dataset('Validation', transforms_train, transforms_test, parse_label_func, annotation_file)
    return testset

class Temporal_Smoother_Trainer(object):
    def __init__(self, model, task, annotation_file, save_path, alpha_list = [7, 8, 9, 10]):
        self.model = model
        self.task = task
        self.annotation_file = annotation_file
        self.alpha_list = alpha_list
        self.save_path = save_path
        self.datasets = self.create_datasets()
        self.three_CV()
    def create_datasets(self):
        data = pickle.load(open(self.annotation_file, 'rb')) # 'Test': 'video_name': video_df
        datasets = []
        for video_name in data['Validation'].keys():
            dataset = get_Dataset_Val({'Validation': {video_name:  data['Validation'][video_name]}},
                train_transforms(299), test_transforms(299), parse_emotion_label)
            datasets.append(dataset)
        return datasets
    def three_CV(self):
        len_videos = len(self.datasets)
        datasets_splits = self.datasets[:len_videos//3], self.datasets[len_videos//3:2*len_videos//3], self.datasets[2*len_videos//3:]
        lines = []
        for train_ids, val_id in zip([[1, 2], [0, 2], [0, 1]], [0, 1, 2]):
            val_dataset = datasets_splits[val_id]
            train_datasets = datasets_splits[train_ids[0]] + datasets_splits[train_ids[1]]
            line = "Task {} {} th Fold".format(self.task, val_id)
            print(line)
            lines.append(line+'\n')
            
            total_metric_alpha = self.evaluate_on_datasets(train_datasets, self.alpha_list)
            for alpha in self.alpha_list:
                line = "alpha : {:.2f} metric: {:.6f}".format(alpha, total_metric_alpha[alpha])
                print(line)
                lines.append(line+'\n')
            index = np.argmax([total_metric_alpha[alpha] for alpha in self.alpha_list])
            best_alpha = self.alpha_list[index]
            val_metric = self.evaluate_on_datasets(val_dataset, [best_alpha])
            line = 'Fold {}, val metric with the best alpha ({:.2f}): {:.6f}\n'.format(val_id, best_alpha, val_metric[best_alpha])
            print(line)
            lines.append(line)
        with open(self.save_path, 'w') as f:
            for l in lines:
                f.write(l)

           
    def evaluate_on_datasets(self, total_datasets, alpha_list):
        
        total_estimates, total_labels = {}, {}
        i = 0
        for video_dataset in total_datasets:
            estimates, labels = self.predict_single_video(video_dataset, alpha_list)
            for alpha in alpha_list:
                if alpha not in total_estimates:
                    total_estimates[alpha] = []
                    total_labels[alpha] = []
                total_estimates[alpha].append(estimates[alpha])
                total_labels[alpha].append(labels)
            i+=1
            # if i==3:
            #     break
        for alpha in alpha_list:
            total_estimates[alpha] = np.concatenate(total_estimates[alpha], axis=0)
            total_labels[alpha] = np.concatenate(total_labels[alpha], axis=0)
        return dict([(alpha, return_metric(total_estimates[alpha], total_labels[alpha], self.task)) for alpha in alpha_list])

    def predict_single_video(self, video_dataset, alpha_list):
        test_dataloader = torch.utils.data.DataLoader(
                    video_dataset,
                    batch_size= 128,
                    shuffle= False,
                    num_workers=8,
                    drop_last=False)
        
        estimates, labels = self.test_one_video(test_dataloader, alpha_list, task = self.task)
        return estimates, labels

    def test_one_video(self, data_loader, alpha_list, task = 'AU'):
        track_metrics = []
        track_labels = []
        total_estimates = {}
        hiddens = None
        with torch.no_grad():
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # evaluate model
                images, labels = val_batch
                outputs, metrics = self.model.forward(images.cuda())
                track_metrics.append(metrics[task].detach().cpu())
                track_labels.append(labels.numpy())
            track_metrics= torch.cat(track_metrics, dim=0)
            i_classifier = self.model.tasks.index(task)
            
            for alpha in alpha_list:
                # get the metrics after temporal smooth: X_t = (\alphaX_{t-1} + mu_t)/(1+alpah)
                track_metrics_alpha = temporal_smooth_function(track_metrics, alpha)
                preds_after_smooth = self.model.emotion_classifiers[i_classifier](track_metrics_alpha.cuda())
                o_task = preds_after_smooth.cpu()
                if task == 'AU':
                    o = (torch.sigmoid(o_task.detach().cpu())>0.5).type(torch.LongTensor)
                    estimates = o.numpy() 
                elif task == 'EXPR':
                    o = F.softmax(o_task.detach().cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                    estimates = o.numpy()
                elif task == 'VA':
                    estimates = o_task.detach().cpu().numpy()
                total_estimates[alpha] = estimates
        track_labels = np.concatenate(track_labels, axis=0)

        return total_estimates, track_labels

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--ckp-save-dir', type=str, default='Temporal_MultiEmotionNet')
    parser.add_argument('--exp-name', type=str, default='experiment_1')
    parser.add_argument('--find_best_lr', action="store_true")
    parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--ckp', type=str, default=None,
        help='if set resume ckp, load it before evaluation')
    parser.add_argument('--avg_features', action = 'store_true',
        help='When true, the model architecture averages the features for EXPR-VA,otherwise, it averages the predictions.')

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

    model = Multitask_EmotionNet('inception_v3', ['AU', 'EXPR', 'VA'], 
        au_names_list, emotion_names_list, va_dim=1, AU_metric_dim = 16,
        lr = args.lr, AU_cls_loss_func = classification_loss_func,
        EXPR_cls_loss_func = cross_entropy_loss,
        VA_cls_loss_func = regression_loss_func,
        AU_metric_loss_func = None,
        EXPR_metric_loss_func = None, 
        VA_metric_loss_func = None, 
        wd=0,
        avg_features = True)
    model.load_state_dict(torch.load(args.ckp)['state_dict'])
    print("load checkpoint from {}".format(args.ckp))
    model.eval()
    model.cuda()

    img_size = 299
    annotation_files = ['create_annotation_file/AU/AU_annotations.pkl',
    'create_annotation_file/EXPR/EXPR_annotations.pkl',
    'create_annotation_file/VA/VA_annotations.pkl']
    tasks = ['AU', 
    'EXPR', 
    'VA']
    save_txt = 'three_cv_res_{}.txt'
    for task, annotation_file in zip(tasks, annotation_files):
        Temporal_Smoother_Trainer(model, task, annotation_file, save_txt.format(task))
