# Improving Few-shot Learning for Talking Face System with TTS Data Augmentation
## Statements
- This repository is only used for academic research, any commercial use is prohibited.
- The copyright of digital human presented in our demo is reserved by SMT.

## Acknowledgements
- Thanks to Shanghai Media Tech(SMT) for providing the data set and rendering service.
- We use pre-trained HuBERT model from [this repository](https://github.com/TencentGameMate/chinese_speech_pretrain).
- We use implementation of soft-DTW loss from [this repository](https://github.com/keonlee9420/Soft-DTW-Loss).
- We use implementation of Transformer from [this repository](https://github.com/Xflick/EEND_PyTorch)

Thanks to the authors of above repositories.

## Demos
### TTS Data Augmentation

https://user-images.githubusercontent.com/86870625/213614533-39d8d2e6-d1bc-45de-a1b7-e9ac6296bb55.mp4

https://user-images.githubusercontent.com/86870625/197924350-01db87eb-b645-4dd4-9c04-2d682730d961.mp4

https://user-images.githubusercontent.com/86870625/197924359-11c29877-c9bd-4c4e-9317-d592f495f91a.mp4

https://user-images.githubusercontent.com/86870625/197924364-4681af6a-a18a-4b9e-9fef-4b2eae43e334.mp4

### TTS-driven Talking Face

https://user-images.githubusercontent.com/86870625/197924168-97cb77a3-1284-4600-bd7f-9f79f2c222d5.mp4

https://user-images.githubusercontent.com/86870625/213614790-8d085cb0-f4c3-4cf8-b3f3-3e748064835d.mp4

https://user-images.githubusercontent.com/86870625/197924196-0ade4a9d-0f36-4199-a7e1-7399215769e6.mp4

https://user-images.githubusercontent.com/86870625/197924206-6b5f3b07-89c0-4d08-b7a0-42590fc35587.mp4

### Different Audio Features

https://user-images.githubusercontent.com/86870625/213614639-21115351-e2a6-4e17-bc96-b1816071423a.mp4

### Different Loss Functions

https://user-images.githubusercontent.com/86870625/197924415-b70030e8-0249-43e1-8c23-38ad4c2f318d.mp4

### Different Data Resources

https://user-images.githubusercontent.com/86870625/197924456-d1c45314-df12-4ee5-957a-3a13f24d1bda.mp4

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
cd ..
cp hubert.py ./fairseq/fairseq/models/hubert/
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
- feature_dir: hubert_large | mfcc | ppg
- train_speaker_list: assign several speaker names for training
- train_json: used to change data resource, path of json file which includes list of audio name in training set
- freq: 50 if feature is chinese_hubert_large or ppg , 100 if feature is mfcc
- input_dim: 39 for mfcc, 128 for ppg

## Validate
run ```bash validate.sh``` to pick the best model by validating on validation set of certain speaker, change ```--val_speaker``` to decide speaker for validation.

## Test
run ```bash test.sh``` to test

## Citation

``` latex
@article{chen2023improving,
  title={Improving Few-Shot Learning for Talking Face System with TTS Data Augmentation},
  author={Chen, Qi and Ma, Ziyang and Liu, Tao and Tan, Xu and Lu, Qu and Yu, Kai and Chen, Xie},
  booktitle={ICASSP 2022-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023}
}
```
