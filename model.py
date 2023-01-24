import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

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
        self.hidden_size = args.hidden_size
        self.use_transformer = args.transformer

        if not args.transformer:
            self.seq2seq = getattr(nn, args.rnn_type)(
                input_size = args.feature_dim,
                hidden_size = self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                )
            self.final_proj = nn.Linear(self.hidden_size*2 if self.bidirectional else self.hidden_size, 37)
            
        else:
            self.seq2seq = TransformerModel(
                in_size = args.feature_dim,
                n_heads = args.n_heads,
                n_units = self.hidden_size,
                n_layers = self.num_layers,
                dim_feedforward = args.dim_feedforward
            )
            self.final_proj = nn.Linear(self.hidden_size, 37)
            
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

        # outputs, (h, c) = self.seq2seq(hidden_states, (h, c))
        if self.use_transformer:
            outputs = self.seq2seq(hidden_states)
        else:
            outputs, _ = self.seq2seq(hidden_states)
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
        
        # outputs, (h, c) = self.seq2seq(hidden_states, (h, c))
        if self.use_transformer:
            outputs = self.seq2seq(hidden_states)
        else:
            outputs, _ = self.seq2seq(hidden_states)
        outputs = self.final_proj(outputs)

        return outputs
    
class TransformerModel(nn.Module):
    def __init__(self, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=True):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.size(1):
                mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)

        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, self.src_mask)
        # output: (B, T, E)
        output = output.transpose(0, 1)
        
        # output: (B, T, C)
        # output = self.decoder(output)

        # if activation:
        #     output = activation(output)

        # output = [out[:ilen] for out, ilen in zip(output, ilens)]

        return output


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)