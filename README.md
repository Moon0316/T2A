# T2A: Robust Text-to-Animation
## Pre-trained model and tools preparation
### Download pre-trained HuBERT model
The pre-trained HuBERT model is obtained from [this repository](https://github.com/TencentGameMate/chinese_speech_pretrain).

Please download Chinese HuBERT model and put it on directory ./data/pretrained_models/ by executing the following command:

```wget -P ./data/pretrained_models/ https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt```

### Download fairseq tool
```
git clone git@github.com:facebookresearch/fairseq.git
cd fairseq
git checkout acd9a53
pip install --editable ./
```

## Feature extraction
### Extract HuBERT feature
```python utils/generate_hubert.py --input_dir ./data/wavs/[speaker name] --output_dir ./data/wav_features/[speaker name]```

### Extract MFCC feature
```python utils/generate_mfcc.py --input_dir ./data/wavs/[speaker name] --output_dir ./data/wav_features/[speaker name]```

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