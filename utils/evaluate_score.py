import csv
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--predict_dir",type=str)
parser.add_argument("--truth_dir",type=str,default='data_ssd/wavs/org/caocao_0912/audio2face_data_for_test_smg_10.7min')
args = parser.parse_args()

import numpy as np
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error

# 61点blendshape
# keys = ['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight',
#         'EyeWideRight', 'JawForward', 'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
#         'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
#         'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch',
#         'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll']

# 52点blendshape
# keys = ['EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight',
#         'EyeWideRight', 'JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
#         'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
#         'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut']

# 52减去眼睛、舌头blendshape
keys = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
        # 'MouthUpperUpRight','CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']

# 读取预测文件和参考答案文件
predict_dir = args.predict_dir
truth_dir = args.truth_dir

n = 0
dtw_score = 0
mse_score = 0
txt_path = os.path.join(predict_dir, 'score.txt')
with open(txt_path, 'w') as f_result:
    for prediction in os.listdir(predict_dir):
        if not prediction.endswith('.csv'):
            continue

        pred = pd.read_csv(os.path.join(predict_dir, prediction))
        pred_data = pred[keys].values
        
        # for test data
        true= pd.read_csv(os.path.join(truth_dir, prediction))
        # for val data
        # true= pd.read_csv(os.path.join(truth_dir, prediction[:-4].upper()+'_anim.csv'))
        true_data = true[keys].values
        n += 1
        # dtw distance
        dtw_distance, _ = fastdtw(true_data, pred_data, dist=mean_squared_error)
        
        # mse distance
        if pred_data.shape[0]<true_data.shape[0]:
            pred_data = np.concatenate((pred_data, np.zeros((true_data.shape[0]-pred_data.shape[0], pred_data.shape[1]))), axis=0)
        else:
            pred_data = pred_data[:true_data.shape[0], :]
        mse_distance = mean_squared_error(true_data, pred_data)
        
        dtw_score += dtw_distance
        mse_score += mse_distance
        print("{} : dtw distance: {}, mse distance: {}".format(prediction, dtw_distance, mse_distance))
        f_result.write("{} : dtw distance: {}, mse distance: {}\n".format(prediction, dtw_distance, mse_distance))
    print('dtw_score: {}\tmse_score: {}'.format(dtw_score/n, mse_score/n))
    f_result.write('dtw_score: {}\tmse_score: {}'.format(dtw_score/n, mse_score/n))
