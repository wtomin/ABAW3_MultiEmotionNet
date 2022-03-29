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
parser.add_argument("--txt_file", type=str, default='/media/Samsung/ABAW3_Challenge/CVPR_2022_3rd_ABAW_Competition _Test_Set_Release_&_Submission_Instructions/Valence_Arousal_Estimation_Challenge_test_set_release.txt')
args = parser.parse_args()

def read_video_list(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines
if __name__=='__main__':
    save_path = 'VA_test_set.pkl'
    if not os.path.exists(save_path):
        data_file = {}
        set_lists = ['Test']
        for set_name in set_lists:
            data_file[set_name] = {}

            for video_name in read_video_list(args.txt_file):
                video_image_dir = os.path.join(args.image_dir, video_name)
                video_frames = sorted(glob.glob(os.path.join(video_image_dir, '*.jpg')))
                video_df = {'path':video_frames, 'frame_id': [int(os.path.basename(frame).split('.')[0]) for frame in video_frames]}
                data_file[set_name][video_name] = pd.DataFrame.from_dict(video_df)
        pickle.dump(data_file, open(save_path, 'wb'))
    else:
        data_file = pickle.load(open(save_path, 'rb'))