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
Expr_list = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise','Other']
plot_Expr_list = [0, 1, 2, 3, 4, 5, 6, 7]
def plot_distribution(all_samples):
    histogram = np.zeros(len(plot_Expr_list))
    for i in range(len(plot_Expr_list)):
        find_true = sum(all_samples==i)
        histogram[i] =find_true/all_samples.shape[0]
    histogram = histogram[plot_Expr_list]
    Expr_list_new = [Expr_list[ii] for ii in plot_Expr_list]
    plt.bar(np.arange(len(Expr_list)), histogram)
    plt.xticks(np.arange(len(Expr_list)), Expr_list_new)
    plt.show()
    print(histogram.max()/histogram)

parser = argparse.ArgumentParser(description='save annotations')
parser.add_argument('--vis', action = 'store_true', 
                    help='whether to visualize the distribution')
parser.add_argument('--annot_dir', type=str, default = '/media/Samsung/ABAW3_Challenge/3rd ABAW Annotations/Third ABAW Annotations/EXPR_Classification_Challenge',
                    help='annotation dir')
parser.add_argument("--image_dir", type=str, default= '/media/Samsung/ABAW3_Challenge/cropped_aligned')

args = parser.parse_args()

def read_txt(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        lines = [l.strip() for l in lines]
    EXPRs = [int(l) for l in lines]
    return np.array(EXPRs)

def parse_video_data(video_name, video_frames, task_label, data_dict):
    data_dict[video_name] = {'path':video_frames, 'frame_id': [int(os.path.basename(frame).split('.')[0]) for frame in video_frames],
    'label': task_label}
    data_dict[video_name] = pd.DataFrame.from_dict(data_dict[video_name])

if __name__=='__main__':
    save_path = 'EXPR_annotations.pkl'
    if not os.path.exists(save_path):
        annot_dir = args.annot_dir
        data_file = {}
        set_lists = ['Train', 'Validation']
        for set_name in set_lists:
            data_file[set_name] = {}
            annotation_files = glob.glob(os.path.join(annot_dir, set_name+"_Set", "*.txt")) 
            for annot_file in tqdm(annotation_files, total=len(annotation_files)):
                EXPRs = read_txt(annot_file)
                video_name = os.path.basename(annot_file).split('.')[0]
                video_image_dir = os.path.join(args.image_dir, video_name)
                paths = sorted(glob.glob(os.path.join(video_image_dir, '*.jpg')))
                if len(paths) != len(EXPRs):
                    N = np.abs(len(paths) - len(EXPRs))
                    print("length mismatch! {} images.".format(N))
                # remove (1) invalid EXPR labels (2) non-existent images
                ids_list = [i for i, x in enumerate(EXPRs) if -1 != x]
                new_paths = []
                new_ids_list = []                                                                                                                                                                                                                                                                    
                for id in ids_list:
                    frame_path = os.path.join(video_image_dir, "{:05d}.jpg".format(id+1))
                    if os.path.exists(frame_path):
                        new_paths.append(frame_path)
                        new_ids_list.append(id)
                paths = new_paths
                EXPRs = EXPRs[new_ids_list]                                                                                                                                                                                                                                                               
                assert len(paths) == len(EXPRs)
                # EXPRs
                parse_video_data(video_name, paths, EXPRs, data_file[set_name])
        pickle.dump(data_file, open(save_path, 'wb'))
    else:
        data_file = pickle.load(open(save_path, 'rb'))
    if args.vis:
        total_dict = {**data_file['Train'], **data_file['Validation']}
        all_samples = np.concatenate([total_dict[x]['label'] for x in total_dict.keys()], axis=0)
        plot_distribution(all_samples)

