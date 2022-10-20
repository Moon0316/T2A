# T2A: Robust Text-to-Animation
## Statements
- This repository is only used for academic research, any commercial use is prohibited.
- The copyright of digital human presented in our demo is reserved by SMT.

## Acknowledgements
- Thanks to Shanghai Media Tech(SMT) for providing the data set and rendering service.
- We use pre-trained HuBERT model from [this repository](https://github.com/TencentGameMate/chinese_speech_pretrain).
- We use implementation of soft-DTW loss from [this repository](https://github.com/keonlee9420/Soft-DTW-Loss).
## Demos
### Text-to Animation
#### org
https://user-images.githubusercontent.com/86870625/196371327-c7a3d161-04fc-49be-ba80-cf55fc1b6101.mp4

#### TTS
https://user-images.githubusercontent.com/86870625/196371512-43fa3e1c-3d8b-4fe5-b1ea-14094cb08ae6.mp4



### TTS Data Augmentation
#### org
https://user-images.githubusercontent.com/86870625/196371687-65b1fb82-290c-49ec-aea8-4e9362dbec2b.mp4


#### TTS Augmentation
https://user-images.githubusercontent.com/86870625/196371713-7196058c-bb25-40f0-8d69-ad77a43e0557.mp4


### Different Features
#### MFCC
https://user-images.githubusercontent.com/86870625/196371842-d23e569e-21f0-4c17-87fe-8411e7dd251f.mp4

#### PPGs
https://user-images.githubusercontent.com/86870625/196371895-e008d60a-29d9-4bcc-a3d0-a5ccd49d468a.mp4

#### HuBERT
https://user-images.githubusercontent.com/86870625/196371957-c9b6b78a-c312-4991-87ad-137d9b1ee515.mp4

### Different Loss Functions
#### L1
https://user-images.githubusercontent.com/86870625/196372191-976bc74c-803b-402c-8d2e-349f874fffa4.mp4

#### L2
https://user-images.githubusercontent.com/86870625/196372230-e5b0b977-4c31-48df-82ce-200deb197c68.mp4

#### soft-DTW
https://user-images.githubusercontent.com/86870625/196372251-e1681d50-93e4-4fa4-b490-c15000aff5c7.mp4

## Pre-trained model and tools preparation
### Download pre-trained HuBERT model
The pre-trained HuBERT model is obtained from [this repository](https://github.com/TencentGameMate/chinese_speech_pretrain).

Please download Chinese HuBERT model and put it on directory ./data/pretrained_models/ by executing the following command:

```
wget -P ./data/pretrained_models/ https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt
```

### Download fairseq tool
```
git clone git@github.com:facebookresearch/fairseq.git
cd fairseq
git checkout acd9a53
pip install --editable ./
```

## Feature extraction
### Extract HuBERT feature
```
python utils/generate_hubert.py --input_dir ./data/wavs/[speaker name] --output_dir ./data/wav_features/[speaker name]
```

### Extract MFCC feature
```
python utils/generate_mfcc.py --input_dir ./data/wavs/[speaker name] --output_dir ./data/wav_features/[speaker name]
```

## Train
run ```bash train.sh``` to train

### important arguments for ```main.py```
- arch: chinese_hubert_large | mfcc | pgg
- feature_combine: True if you want to use weighted sum of hubert feature
- output_path: "result" if you want to generate output of test set | [other name] if you want to generate other data
- test_input_path: you should explicitly assign path of test_input_path if output_path != "result", test_input_path is the dir of csv files
- test_epoch: do not need to explicitly assign, will find the model with best 
- root_dir: dir of dataset root
- feature_dir: hubert_large | wav2vec2_large | mfcc | ppg
- train_speaker_list: assign several speaker names for training
- train_json: used to change data resource, path of json file which includes list of audio name in training set
- freq: 50 if feature is chinese_hubert_large or ppg , 100 if feature is mfcc
- input_dim: 39 for mfcc, 128 for ppg

## Test
run ```bash test.sh``` to test

## Validate
run ```bash validate.sh``` to pick the best model by validating on validation set of certain speaker, change ```--val_speaker``` to decide speaker for validation.
