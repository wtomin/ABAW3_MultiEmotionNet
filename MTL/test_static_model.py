import os
import cv2
import glob
import pickle
import numpy as np
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.static_linear_model import Multitask_EmotionNet
from data.dataset import ImageSequenceDataset, ImageDataset
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
def parse_none_label(df_row):
    return 0
def refill_frames_ids(frames_ids, length):
    sample_index = []
    prev_sample_index = 0
    for frame_index in range(1, length+1):
        if len(np.where(frames_ids==frame_index)[0])==1:
            idx = np.where(frames_ids==frame_index)[0][0]
            sample_index.append(idx)
            prev_sample_index = idx
        else:
            sample_index.append(prev_sample_index)
    return sample_index

def get_Dataset_Test(annotation_file, video = False, transforms_train=None, transforms_test=None, 
    parse_label_func=parse_none_label,
    seq_len = None):
    if video:
        assert seq_len is not None, "seq_len needs specification"
        Dataset = ImageSequenceDataset
        testset = Dataset('Test', seq_len, transforms_train, transforms_test, parse_label_func, annotation_file)
    else:
        Dataset = ImageDataset
        testset = Dataset('Test', transforms_train, transforms_test, parse_label_func, annotation_file)
    return testset

class Tester_Single_task(object):
    def __init__(self, model, task, annotation_file, save_dir, video_dir, video=False, seq_len = None):
        self.model = model
        self.task = task
        self.annotation_file = annotation_file
        self.save_dir = save_dir
        self.video_dir = video_dir
        self.video = video
        self.seq_len = seq_len
        self.video_datasets = self.create_video_datasets()
        self.predict_multiple_videos()
    def create_video_datasets(self):
        data = pickle.load(open(self.annotation_file, 'rb')) # 'Test': 'video_name': video_df
        video_datasets = {}
        for video_name in data['Test'].keys():
            video_datasets[video_name] = get_Dataset_Test({'Test': {video_name:  data['Test'][video_name]}},
                self.video, train_transforms(299), test_transforms(299), parse_none_label, self.seq_len)
        return video_datasets

    def predict_multiple_videos(self):
        for video_name in tqdm(self.video_datasets):
            save_path = '{}/{}.txt'.format(self.task, video_name)
            video_dataset = self.video_datasets[video_name]
            track, frames_ids = self.predict_single_video(video_dataset, video_name)
            #filtered out repeated frames
            mask = np.zeros_like(frames_ids, dtype=bool)
            mask[np.unique(frames_ids, return_index=True)[1]] = True
            frames_ids = frames_ids[mask]
            track = track[mask]

            # check the video real frame length
            def matched_video_name(query, target):
                query = query.replace('_left', '') 
                query = query.replace('_right', '')
                if query == target:
                    return True
                else:
                    return False
            video_path = [v_p for v_p in videos_list if matched_video_name(video_name, v_p.split('/')[-1].split('.')[0])]
            assert len(video_path) ==1
            video_path = video_path[0]
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(7)) + 1

            # find out the missing frames and fill in with previous frames
            sample_index = refill_frames_ids(frames_ids, length)
            predictions = track[np.array(sample_index)]
            self.save_to_file(predictions, save_path, task=self.task)
            

    def predict_single_video(self, video_dataset, video_name):
        test_dataloader = torch.utils.data.DataLoader(
                    video_dataset,
                    batch_size= 32,
                    shuffle= False,
                    num_workers=8,
                    drop_last=False)
        
        track = self.test_one_video(test_dataloader, task = self.task)
        frames_ids = video_dataset.df['frame_id'].values
        assert len(track) == len(frames_ids)
        return track, frames_ids

    def test_one_video(self, data_loader, task = 'AU'):
        track_val = []
        hiddens = None
        with torch.no_grad():
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # evaluate model
                images, labels = val_batch
                outputs, _ = self.model.forward(images.cuda())
                track_val.append(self._format_estimates(outputs)[task])
        track_val= np.concatenate(track_val, axis=0)

        return track_val
    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                o_task = output[task]
                o = (torch.sigmoid(o_task.detach().cpu())>0.5).type(torch.LongTensor)
                estimates['AU'] = o.numpy() 
            elif task == 'EXPR':
                o_task = output[task] if self.model.avg_features else output[task].mean(-2)
                o = F.softmax(o_task.detach().cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                o_task = output[task] if self.model.avg_features else output[task].mean(-2)
                estimates['VA'] = o_task.detach().cpu().numpy()

        return estimates
    def save_to_file(self, predictions, save_path, task= 'AU'):
        save_path =os.path.join(self.save_dir, save_path)
        save_dir = os.path.dirname(os.path.abspath(save_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        categories = PRESET_VARS.ABAW3.categories[task]
        with open(save_path, 'w') as f:
            f.write(",".join(categories)+"\n")
            for i, line in enumerate(predictions):
                if isinstance(line, np.ndarray):
                    digits = []
                    for x in line:
                        if isinstance(x, np.float32):
                            digits.append("{:.4f}".format(x))
                        elif isinstance(x, np.int64):
                            digits.append(str(x))
                    line = ','.join(digits)+'\n'
                elif isinstance(line, np.int64):
                    line = str(line)+'\n'
                # if i == len(predictions)-1:
                #     line = line[:-1]
                f.write(line)

class Tester_MTL(Tester_Single_task):
    def __init__(self, model,  annotation_file, save_dir, video=False, seq_len = None):
        self.model = model
        self.annotation_file = annotation_file
        self.save_dir = save_dir
        self.video = video
        self.seq_len = seq_len
        self.video_datasets = self.create_video_datasets()
        self.predict_multiple_videos()

    def predict_single_video(self, video_dataset, video_name):
        test_dataloader = torch.utils.data.DataLoader(
                    video_dataset,
                    batch_size= 32,
                    shuffle= False,
                    num_workers=8,
                    drop_last=False)

        track_multitasks = self.test_one_video(test_dataloader)
        frames_paths = video_dataset.df['path'].values
        assert len(track_multitasks['AU']) == len(frames_paths)
        return track_multitasks, frames_paths

    def test_one_video(self, data_loader):
        
        tasks = ['AU', 'EXPR', 'VA']
        track_val = dict([(t, []) for t in tasks])
        with torch.no_grad():
            for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
                # evaluate model
                images, labels = val_batch
                outputs, _ = self.model.forward(images.cuda())
                estimates = self._format_estimates(outputs)
                for t in tasks:
                    track_val[t].append(estimates[t])
        for t in track_val:
            track_val[t]= np.concatenate(track_val[t], axis=0)
        return track_val
    def parse_video_predictions_into_string_lines(self, track_preds, frames_paths):
        lines = []
        for i in range(len(frames_paths)):
            line = "{},".format(frames_paths[i])
            for t in ['VA',  'EXPR', 'AU']:
                pred = track_preds[t][i]
                if t == 'VA':
                    line += "{:.4f},{:.4f},".format(pred[0], pred[1])
                elif t =='EXPR':
                    line += "{:d},".format(pred)
                else:
                    line += ','.join(['{:d}'.format(pred[j]) for j in range(12)])
            line+='\n'
            lines.append(line)
        return lines

    def predict_multiple_videos(self):
        lines = ['image,valence,arousal,expression,aus\n']
        for video_name in tqdm(self.video_datasets):
            video_dataset = self.video_datasets[video_name]
            track_multitasks, frames_paths = self.predict_single_video(video_dataset, video_name)
            frames_paths = ['/'.join(f.split('/')[-2:]) for f in frames_paths]
            
            video_lines = self.parse_video_predictions_into_string_lines(track_multitasks, frames_paths)
            lines.extend(video_lines)
        self.save_to_file('MTL_predictions.txt', lines)
    def save_to_file(self, save_path, save_lines):
        save_path =os.path.join(self.save_dir, save_path)
        save_dir = os.path.dirname(os.path.abspath(save_path))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w') as f:
            for line in save_lines:
                f.write(line)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--ckp-save-dir', type=str, default='Static_MultiEmotionNet')
    # parser.add_argument('--video', type=bool, action="store_true", help='whether the data is image sequence')
    parser.add_argument('--video_dir', type=str, default = '/media/Samsung/Aff-wild2-Challenge/videos',
        help='the video directory used to get the frame length of each video')
    # parser.add_argument('--find_best_lr', action="store_true")
    # parser.add_argument('--lr', type=float, default = 1e-3)
    parser.add_argument('--ckp', type=str, default=None,
        help='if set resume ckp, load it before evaluation')
    parser.add_argument('--save_dir', type=str, default = 'save_static_predictions')
    parser.add_argument('--avg_features', action = 'store_true',
        help='When true, the model architecture averages the features for EXPR-VA,otherwise, it averages the predictions.')

    args = parser.parse_args()
    au_names_list = PRESET_VARS.ABAW3.categories['AU']
    emotion_names_list = PRESET_VARS.ABAW3.categories['EXPR']
    videos_list = glob.glob(os.path.join(args.video_dir, 'batch1', '*.mp4')) +  glob.glob(os.path.join(args.video_dir, 'batch2', '*.mp4'))
    videos_list += glob.glob(os.path.join(args.video_dir, 'batch1', '*.avi')) +  glob.glob(os.path.join(args.video_dir, 'batch2', '*.avi'))
    model = Multitask_EmotionNet('inception_v3', ['AU', 'EXPR', 'VA'], 
        au_names_list, emotion_names_list, va_dim=1, AU_metric_dim = 16,
        lr = 0.001, AU_cls_loss_func = None,
        EXPR_cls_loss_func = None,
        VA_cls_loss_func = None,
        AU_metric_loss_func = None, #AU_metric_loss,
        EXPR_metric_loss_func = None, #EXPR_metric_loss,
        VA_metric_loss_func = None, #VA_metric_loss,
        wd=0,
        avg_features = args.avg_features)
    model.load_state_dict(torch.load(args.ckp)['state_dict'])
    print("Loaded from the checkpoint {}".format(args.ckp))
    model.eval()
    model.cuda()

    img_size = 299

    # dataset 1
    annotation_files = ['create_annotation_file/AU/AU_test_set.pkl',
    'create_annotation_file/EXPR/EXPR_test_set.pkl',
    'create_annotation_file/VA/VA_test_set.pkl']
    tasks = ['AU', 
    'EXPR', 
    'VA']
    
    for task, annotation_file in zip(tasks, annotation_files):
        Tester_Single_task(model, task, annotation_file, args.save_dir, args.video_dir)

    annotation_file = 'create_annotation_file/MTL/MTL_test_set.pkl'
    Tester_MTL(model, annotation_file, args.save_dir)

    