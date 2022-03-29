import os
import pickle
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import matplotlib
font = {
        'size'   : 25}
matplotlib.rc('font', **font)
from tqdm import tqdm
import glob
np.random.seed(0)
import warnings

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action = 'store_true', 
                    help='whether to visualize the distribution')
parser.add_argument('--annot_dir', type=str, default = '/media/Samsung/ABAW3_Challenge/3rd ABAW Annotations/Third ABAW Annotations/MTL_Challenge',
                    help='annotation dir')
parser.add_argument("--image_dir", type=str, default= '/media/Samsung/ABAW3_Challenge/cropped_aligned')

args = parser.parse_args()
AUs_name_list = ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26']
Expr_list = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise','Other']
plot_Expr_list = [0, 1, 2, 3, 4, 5, 6, 7]

def read_txt(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        lines = [l.strip() for l in lines]
        lines = [l.split(',') for l in lines]
    path = [l[0] for l in lines]
    valence = [float(l[1]) for l in lines]
    arousal = [float(l[2]) for l in lines]
    expr = [int(l[3]) for l in lines]
    AUs = [np.array([float(x) for x in l[4:]]) for l in lines]
    return path, np.stack([valence, arousal], axis=-1), np.array(expr), np.array(AUs)

def parse_video_data(video_name, video_frames, AU_label, EXPR_label, VA_label, data_dict):
    data_dict[video_name] = {'path':video_frames, 'frame_id': [int(os.path.basename(frame).split('.')[0]) for frame in video_frames]}
    if AU_label is not None:
        for i_AU, AU_name in enumerate(AUs_name_list):
            data_dict[video_name][AU_name] = AU_label[:, i_AU]
    if EXPR_label is not None:
        data_dict[video_name]['label'] = EXPR_label
    if VA_label  is not None:
        data_dict[video_name]['valence'] = VA_label[:, 0]
        data_dict[video_name]['arousal'] = VA_label[:, 1]
    data_dict[video_name] = pd.DataFrame.from_dict(data_dict[video_name])

def parse_video_name(frame_paths):
    videos = [f.split('/')[-2] for f in frame_paths]
    videos, indexes = np.unique(videos, return_inverse=True)
    video_ids_dict = {}
    for i, v in enumerate(videos):
        mask = indexes == i
        video_ids_dict[v] = np.arange(len(mask))[mask] 
    return video_ids_dict
def get_val_videos_set(task_names):
    task_names = task_names.split("_")
    val_videos = []
    for task in task_names:
        if task=='AU':
            file = '../AU/AU_annotations.pkl'
        elif task =='EXPR':
            file = '../EXPR/EXPR_annotations.pkl'
        elif task=='VA':
            file = '../VA/VA_annotations.pkl'
        data = pickle.load(open(file, 'rb'))
        val_videos.extend(list(data['Validation'].keys()))
    return val_videos
if __name__=='__main__':
    save_path = 'annotations.pkl'
    if not os.path.exists(save_path):
        annot_dir = args.annot_dir
        annot_files = ['train_set.txt', 'validation_set.txt']
        set_lists = ['Train', 'Validation']
        for name in ['AU_EXPR', 'AU_VA', 'EXPR_VA', 'AU_EXPR_VA']:
            data_file = {}
            val_videos_list_other_datasets = get_val_videos_set(name)
            for annot_file, set_name in zip(annot_files, set_lists):
                data_file[set_name] = {}
                txt_file = os.path.join(args.annot_dir, annot_file)
                paths, VA, expr, AUs = read_txt(txt_file)
                paths = [os.path.join(args.image_dir, x) for x in paths]
                ids_list = [i for i, x in enumerate(paths) if os.path.exists(x)]
                if name=='AU_EXPR':
                    ids_list = [i for i in ids_list if expr[i]!=-1 and -1 not in AUs[i] and -5 in VA[i]]
                elif name=='AU_VA':
                    ids_list = [i for i in ids_list if -5 not in VA[i] and -1 not in AUs[i] and expr[i]==-1]
                elif name=='EXPR_VA':
                    ids_list = [i for i in ids_list if expr[i]!=-1  and -5 not in VA[i] and -1 in AUs[i]]
                elif name == 'AU_EXPR_VA':
                    ids_list = [i for i in ids_list if expr[i]!=-1 and -1 not in AUs[i] and -5 not in VA[i]]
                new_paths = [paths[i] for i in ids_list]
                # parse video
                if len(new_paths)>0:
                    video_names_dict = parse_video_name(new_paths)
                    for video_name in video_names_dict.keys():
                        video_frame_ids = video_names_dict[video_name]
                        if name=='AU_EXPR':
                            new_AU = AUs[ids_list][video_frame_ids]
                            new_expr = expr[ids_list][video_frame_ids]
                            new_VA = None
                        elif name=='AU_VA':
                            new_AU = AUs[ids_list][video_frame_ids]
                            new_expr = None
                            new_VA = VA[ids_list][video_frame_ids]
                        elif name=='EXPR_VA':
                            new_AU = None
                            new_expr = expr[ids_list][video_frame_ids]
                            new_VA = VA[ids_list][video_frame_ids]
                        elif name=='AU_EXPR_VA':
                            new_AU = AUs[ids_list][video_frame_ids]
                            new_expr = expr[ids_list][video_frame_ids]
                            new_VA = VA[ids_list][video_frame_ids]    
                        video_path = [new_paths[i] for i in video_frame_ids]
                        if (set_name=='Train' and video_name not in val_videos_list_other_datasets) or set_name !='Train':
                            parse_video_data(video_name, video_path, new_AU, new_expr, new_VA,
                            data_file[set_name])
            pickle.dump(data_file, open(name+'_'+save_path, 'wb'))
    else:
        data_file = pickle.load(open(save_path, 'rb'))

    # if args.vis:
        # total_dict = data_file
        # arousal_names = ['arousal_{}'.format(i+1) for i in range(6)]
        # valence_names = ['valence_{}'.format(i+1) for i in range(6)]
        # all_samples_valence = np.concatenate([data_file['Train'][valence_names].values, data_file['Validation'][valence_names].values], axis=0)
        # all_samples_arousal = np.concatenate([data_file['Train'][arousal_names].values, data_file['Validation'][arousal_names].values], axis=0)

        # plt.hist2d(all_samples_valence.mean(axis=-1), all_samples_arousal.mean(axis=-1), bins=(20, 20), cmap=plt.cm.jet)
        # plt.xlabel("Valence")
        # plt.ylabel('Arousal')
        # plt.colorbar()
        # plt.show()

