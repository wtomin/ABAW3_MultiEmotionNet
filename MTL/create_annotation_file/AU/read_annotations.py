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
import re
np.random.seed(0)
import warnings

def plot_pie(AU_list, pos_freq, neg_freq):
    ploting_labels = [x+'+ {0:.2f}'.format(y) for x, y in zip(AU_list, pos_freq)] + [x+'- {0:.2f}'.format(y) for x, y in zip(AU_list, neg_freq)] 
    cmap = matplotlib.cm.get_cmap('coolwarm')
    colors = [cmap(x) for x in pos_freq] + [cmap(x) for x in neg_freq]
    fracs = np.ones(len(AU_list)*2)
    plt.pie(fracs, labels=ploting_labels, autopct=None, shadow=False, colors=colors,startangle =78.75)
    plt.title("AUs distribution")
    plt.show()

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action = 'store_true', 
                    help='whether to visualize the distribution')
parser.add_argument('--annot_dir', type=str, default = '/media/Samsung/ABAW3_Challenge/3rd ABAW Annotations/Third ABAW Annotations/AU_Detection_Challenge',
                    help='annotation dir')
parser.add_argument("--image_dir", type=str, default= '/media/Samsung/ABAW3_Challenge/cropped_aligned')

args = parser.parse_args()
AUs_name_list = ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26']
def read_txt(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        lines = [l.strip() for l in lines]
        lines = [l.split(',') for l in lines]
    AUs = [np.array([float(x) for x in l]) for l in lines]
    return np.array(AUs)

def parse_video_data(video_name, video_frames, task_label, data_dict):
    data_dict[video_name] = {'path':video_frames, 'frame_id': [int(os.path.basename(frame).split('.')[0]) for frame in video_frames]}
    assert task_label.shape[-1] == len(AUs_name_list) # Number of AUs needs to be the same
    for i_AU, AU_name in enumerate(AUs_name_list):
        data_dict[video_name][AU_name] = task_label[:, i_AU]
    data_dict[video_name] = pd.DataFrame.from_dict(data_dict[video_name])

if __name__=='__main__':
    save_path = 'AU_annotations.pkl'
    if not os.path.exists(save_path):
        annot_dir = args.annot_dir
        data_file = {}
        set_lists = ['Train', 'Validation']
        for set_name in set_lists:
            data_file[set_name] = {}
            annotation_files = glob.glob(os.path.join(annot_dir, set_name+"_Set", "*.txt")) 

            for annot_file in tqdm(annotation_files, total=len(annotation_files)):
                AUs = read_txt(annot_file)
                video_name = os.path.basename(annot_file).split('.')[0]
                video_image_dir = os.path.join(args.image_dir, video_name)
                paths = sorted(glob.glob(os.path.join(video_image_dir, '*.jpg')))
                if len(paths) != len(AUs):
                    N = np.abs(len(paths) - len(AUs))
                    print("length mismatch! {} images.".format(N))
                # remove (1) invalid AU labels (2) non-existent images
                ids_list = [i for i, x in enumerate(AUs) if -1 not in x]
                new_paths = []
                new_ids_list = []                                                                                                                                                                                                                                                                    
                for id in ids_list:
                    frame_path = os.path.join(video_image_dir, "{:05d}.jpg".format(id+1))
                    if os.path.exists(frame_path):
                        new_paths.append(frame_path)
                        new_ids_list.append(id)
                paths = new_paths
                AUs = AUs[new_ids_list]                                                                                                                                                                                                                                                               
                assert len(paths) == len(AUs)
                # AU
                parse_video_data(video_name, paths, AUs, data_file[set_name])
        pickle.dump(data_file, open(save_path, 'wb'))
    else:
        data_file = pickle.load(open(save_path, 'rb'))
    if args.vis:
        total_dict = {**data_file['Train'], **data_file['Validation']}
        all_samples = np.concatenate([total_dict[x][AUs_name_list] for x in total_dict.keys()], axis=0)
        pos_freq = np.sum(all_samples, axis=0)/all_samples.shape[0]
        neg_freq = -np.sum(all_samples-1, axis=0)/all_samples.shape[0]
        print("pos_weight:", neg_freq/pos_freq)
        plot_pie(AUs_name_list, pos_freq, neg_freq)

