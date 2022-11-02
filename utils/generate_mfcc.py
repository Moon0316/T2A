
import os
import time
import argparse
import librosa
import python_speech_features
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./data/wavs/org')
parser.add_argument('--output_dir', type=str, default='./data/wav_features/org')
args = parser.parse_args()

def handle_wav(wav_path, saved_to_path):
    wav, sr = librosa.load(wav_path, sr=16000)
    input_values = python_speech_features.mfcc(signal=wav,samplerate=sr,numcep=13,winlen=0.025,winstep=0.01)
    d_mfcc_feat = python_speech_features.base.delta(input_values, 1)
    d_mfcc_feat2 = python_speech_features.base.delta(input_values, 2)
    input_values = np.hstack((input_values, d_mfcc_feat, d_mfcc_feat2))
    input_values.dump(saved_to_path)
    print('saved to {}, shape={}'.format(saved_to_path,input_values.shape))

if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    for dataset in os.listdir(args.input_dir):
        os.makedirs(os.path.join(args.output_dir, 'mfcc', dataset), exist_ok=True)
        for file in os.listdir(os.path.join(args.input_dir, dataset)):
            if file.endswith('.wav'):
                wav_path = os.path.join(args.input_dir,dataset,file)
                file_name = file.split('.wav')[0]
                save_path = os.path.join(args.output_dir, 'mfcc', dataset, file_name + '.npy')
                handle_wav(wav_path, save_path)

    end = time.time() - start
    print(end)
