import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.Autoformer_EncDec import moving_avg
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, DSAttention
from layers.Embed import DataEmbedding
import numpy as np
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        configs.d_model = configs.seq_len
        # Embedding

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

        self.proj = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.prov = nn.Linear(configs.dec_in, configs.d_model,bias=True)
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear',nn.Linear(configs.seq_len, self.pred_len))
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin]*configs.enc_in),requires_grad=True)
        self.revin_layer = RevIN(configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_enc = self.revin_layer(x_enc, 'norm')
        # enc_out = x_enc.permute(0, 2, 1)
        enc_out = self.prov(x_enc).permute(0, 2, 1)
        # enc_out = self.proj(enc_out)
        dec_in = self.dec_embedding(x_enc,x_mark_enc)
        dec_out = self.decoder(enc_out,dec_in)
        # dec_out = self.proj(dec_out)
        # dec_out = dec_out.permute(0, 2, 1)
        linear_out = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out, 'denorm')

        if self.output_attention:

            return dec_out[:, -self.pred_len:, :] #,attns
        else:
            return dec_out