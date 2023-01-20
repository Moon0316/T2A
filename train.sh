# arch: 'mfcc','ppg', 'chinese_hubert_large'
# loss_type: L1 L2 SoftDTW
# rnn_type: LSTM GRU RNN
python main.py --arch chinese_hubert_large \
               --name demo \
               --feature_dim 128 \
               --cov_dim 128 \
               --input_dim 39 \
               --seed 999 \
               --loss_type SoftDTW \
               --max_epoch 30 \
               --gamma 0.001 \
               --root_dir ./data \
               --train_blendshape_path ./data/wavs/org/train.pkl \
               --val_blendshape_path ./data/wavs/org/val.pkl \
               --num_workers 8 \
               --train_batch_size  8 \
               --val_batch_size  8 \
               --test_batch_size 1 \
               --gpu 0 \
               --train_json dataset_json/demo.json \
               --train_speaker_list org zh-CN-XiaochenNeural \
               --feature_combine \
               --rnn_type LSTM
