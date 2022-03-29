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

parser = argparse.ArgumentParser(description='save test set')
parser.add_argument("--image_dir", type=str, default= '/media/Samsung/ABAW3_Challenge/cropped_aligned')
parser.add_argument("--txt_file", type=str, default='/media/Samsung/ABAW3_Challenge/CVPR_2022_3rd_ABAW_Competition _Test_Set_Release_&_Submission_Instructions/Multi_Task_Learning_Challenge_test_set_release.txt')
args = parser.parse_args()

def read_video_list(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()[1: ]
        lines = [l.strip() for l in lines]
    return lines
if __name__=='__main__':
    save_path = 'MTL_test_set.pkl'
    if not os.path.exists(save_path):
        data_file = {}
        set_lists = ['Test']
        videos_list = []
        for set_name in set_lists:
            data_file[set_name] = {}
            images = read_video_list(args.txt_file)
            video_names = [image.split('/')[0] for image in images]
            video_frames_dict = {}
            for i_image, video in enumerate(video_names):
                if video not in video_frames_dict.keys():
                    video_frames_dict[video] = []
                    videos_list.append(video)
                image_path  = os.path.join(args.image_dir, images[i_image])
                if os.path.exists(image_path):
                    video_frames_dict[video].append(image_path)
                else:
                    video_frames_dict[video].append(video_frames_dict[video][-1]) # replicate the previous frame
                    print("{} not exist, replaced by {}".format(image_path, video_frames_dict[video][-1]))
            for video_name in video_frames_dict.keys():
                video_frames = video_frames_dict[video_name]
                video_df = {'path':video_frames, 'frame_id': [int(os.path.basename(frame).split('.')[0]) for frame in video_frames]}
                data_file[set_name][video_name] = pd.DataFrame.from_dict(video_df)
        data_file['videos'] = videos_list
        pickle.dump(data_file, open(save_path, 'wb'))
    else:
        data_file = pickle.load(open(save_path, 'rb'))