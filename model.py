import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


from fairseq.models.speech_to_text import Conv1dSubsampler
class EncoderDecoderModel(nn.Module):
    def __init__(self, args):
        super(EncoderDecoderModel, self).__init__()

        self.align = False if args.loss_type == 'SoftDTW' else True
        self.feature_combine = args.feature_combine
        self.cov_dim = args.cov_dim
        self.feature_dim = args.feature_dim
        self.feature_enc_layers = [(args.cov_dim,3,2), (args.cov_dim,3,2)]
        self.gamma = args.gamma
        self.device = args.device

        if args.arch == 'mfcc':
            self.subsample = Conv1dSubsampler(
                in_channels = args.input_dim,
                mid_channels = args.cov_dim,
                out_channels = args.cov_dim,
                kernel_sizes = (3,3)
                )
            self.feature_layernorm = nn.LayerNorm(args.cov_dim)
            self.audio_feature_map = nn.Linear(args.cov_dim, args.feature_dim)
        elif args.arch == 'ppg':
            self.subsample = Conv1dSubsampler(
                in_channels = args.input_dim,
                mid_channels = args.cov_dim,
                out_channels = args.cov_dim,
                kernel_sizes = (3,)
                )
            self.feature_layernorm = nn.LayerNorm(args.cov_dim)
            self.audio_feature_map = nn.Linear(args.cov_dim, args.feature_dim)

        elif args.arch == 'chinese_hubert_large':
            self.subsample = Conv1dSubsampler(
                in_channels = 1024,
                mid_channels = args.cov_dim,
                out_channels = args.cov_dim,
                kernel_sizes = (3,)
                )
            
            self.scale = nn.Parameter(torch.ones(24)/24)
            self.feature_layernorm = nn.LayerNorm(128)
            self.audio_feature_map = nn.Linear(128, args.feature_dim)
        else:
            print('check the arch type!')
            exit(1)

        self.arch = args.arch
        self.num_layers = 1
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = 128

        self.rnn = getattr(nn, args.rnn_type)(
            input_size = args.feature_dim,
            hidden_size = self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            )
        self.final_proj = nn.Linear(self.hidden_size*2 if self.bidirectional else self.hidden_size, 37)

    def forward(self, audio, blendshapes, audio_lengths, blendshape_lengths):

        audio_shape = audio.shape
        batch_size = audio_shape[0]
        if self.arch in ['mfcc','ppg']:   
            hidden_states, input_lengths = self.subsample(audio, torch.tensor(audio_shape[:-1])) # TxBxC
            hidden_states = hidden_states.transpose(0,1) # BxTxC
            hidden_states = self.feature_layernorm(hidden_states) 
            hidden_states = self.audio_feature_map(hidden_states)
        elif self.arch == 'chinese_hubert_large':
            audio = audio.transpose(1,3) #BxFxTx 24 layers
            if self.feature_combine:
                # weighted_feature = F.softmax(self.scale) * audio
                weighted_feature = F.softmax(self.scale) * audio
                weighted_feature = weighted_feature.transpose(1,2)
                weighted_feature = weighted_feature.sum(-1) # B*T*1024
            else:
                weighted_feature = audio[:,:,:,23]
                weighted_feature = weighted_feature.transpose(1,2)

            hidden_states, input_lengths = self.subsample(weighted_feature, torch.tensor(weighted_feature.shape[:-1])) # TxBxC
            hidden_states = hidden_states.transpose(0,1) # BxTxC
            hidden_states = self.feature_layernorm(hidden_states) 
            hidden_states = self.audio_feature_map(hidden_states)
        else:
            print('check the arch type!')
            exit(1)

        # Align audio and blendshape when loss type is not SoftDTW
        if self.align:
            if blendshapes.shape[1] > hidden_states.shape[1]:
                blendshapes = blendshapes[:,:int(hidden_states.shape[1]),:]
            elif blendshapes.shape[1] < hidden_states.shape[1]:
                hidden_states = hidden_states[:,:int(blendshapes.shape[1]),:]
        seq_len = hidden_states.shape[1]
        
        outputs = torch.zeros(batch_size, seq_len, 37).to(device=self.device)
        # h = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(device=self.device)
        # c = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(device=self.device)

        # outputs, (h, c) = self.rnn(hidden_states, (h, c))
        outputs, _ = self.rnn(hidden_states)
        outputs = self.final_proj(outputs)

        if blendshapes is None: # test mode
            return outputs

        return outputs, blendshapes


    def predict(self, audio):
        audio_shape = audio.shape
        if self.arch in ['mfcc','ppg']:   
            hidden_states, input_lengths = self.subsample(audio, torch.tensor(audio_shape[:-1])) # TxBxC
            hidden_states = hidden_states.transpose(0,1) # BxTxC
            hidden_states = self.feature_layernorm(hidden_states) 
            hidden_states = self.audio_feature_map(hidden_states)
            
        elif self.arch == 'chinese_hubert_large':
            audio = audio.transpose(1,3) #BxFxTx 24 layers
            
            if self.feature_combine:
                # weighted_feature = F.softmax(self.scale) * audio
                weighted_feature = F.softmax(self.scale) * audio
                weighted_feature = weighted_feature.transpose(1,2)
                weighted_feature = weighted_feature.sum(-1) # B*T*1024
            else:
                weighted_feature = audio[:,:,:,23]
                weighted_feature = weighted_feature.transpose(1,2)

            hidden_states, input_lengths = self.subsample(weighted_feature, torch.tensor(weighted_feature.shape[:-1])) # TxBxC
            hidden_states = hidden_states.transpose(0,1) # BxTxC
            hidden_states = self.feature_layernorm(hidden_states) 
            hidden_states = self.audio_feature_map(hidden_states)
        else:
            print("check the arch type!")
            exit(1)
            
        seq_len = hidden_states.shape[1]
   
        batch_size = 1
        outputs = torch.zeros(batch_size, seq_len, 37).to(device=self.device)
        # h = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(device=self.device)
        # c = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(device=self.device)
        
        # outputs, (h, c) = self.rnn(hidden_states, (h, c))
        outputs, _ = self.rnn(hidden_states)
        outputs = self.final_proj(outputs)

        return outputs