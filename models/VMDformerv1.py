import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.Crossformer_EncDec import Encoder, Decoder, DecoderLayer
from layers.Autoformer_EncDec import moving_avg
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, DSAttention, TwoStageAttentionLayernoseg
from layers.Embed import DataEmbedding
import numpy as np
from layers.RevIN import RevIN
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid
from math import ceil

class scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10):
        super(scale_block, self).__init__()

        # if win_size > 1:
        #     self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        # else:
        #     self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayernoseg(configs, seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # _, ts_dim, _, _ = x.shape

        # if self.merge_layer is not None:
        #     x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x, None
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]



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
        self.dmerging = 3
        self.seg_num = configs.enc_in
        self.in_seg_num = configs.enc_in
        configs.d_model = configs.seq_len
        # self.win_size = 1
        # # Embedding
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # # Encoder
        # self.encoder = Encoder(
        #     [
        #         scale_block(configs, 1 if l is 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
        #                     1, configs.dropout,
        #                     self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
        #                     ) for l in range(configs.e_layers)
        #     ]
        # )
        # self.encoder = TwoStageAttentionLayernoseg(configs, self.seg_num, configs.factor, configs.d_model,
        #                                        configs.n_heads, configs.d_ff, configs.dropout)
        self.GRU = nn.GRU(self.pred_len, hidden_size=configs.seq_len)

        # self.attent = TwoStageAttentionLayernoseg(configs, self.seg_num, configs.factor, configs.d_model,
        #                                        configs.n_heads, configs.d_ff, configs.dropout)
        self.encoder = nn.Sequential()

        for i in range(2):
            self.encoder.append(nn.LayerNorm(configs.seq_len))
            self.encoder.append(TwoStageAttentionLayernoseg(configs, self.seg_num, configs.factor, configs.d_model,
                                               configs.n_heads, configs.d_ff, configs.dropout))
            self.encoder.append(nn.Dropout(configs.dropout))
            # self.encoder.append(nn.Linear(self.pred_len,configs.seq_len))
            # self.encoder.append(nn.GELU())

            self.encoder.append(nn.Linear(self.pred_len,512))
            self.encoder.append(nn.GELU())
            self.encoder.append(nn.Linear(512,configs.seq_len))
            # self.encoder.append(nn.Dropout(configs.dropout))
        self.decoder = nn.Sequential()
        for i in range(2):
            self.decoder.append(nn.LayerNorm(self.pred_len))
            self.decoder.append(self.GRU)

            self.decoder.append(SelectItem(0))
            self.decoder.append(nn.Linear(configs.seq_len,self.pred_len))
            self.decoder.append(nn.GELU())
            self.decoder.append(nn.Dropout(configs.dropout))
        # self.encoder.append(nn.Dropout(configs.dropout))
        # self.encoder.append(nn.Linear(configs.seq_len,configs.seq_len,bias=True))
        # self.encoder.append(nn.ReLU())

        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #
        #     self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                        configs.dropout)
        #
        #     self.decoder = Decoder(
        #         [
        #             DecoderLayer(
        #                 AttentionLayer(
        #                     FullAttention(True, configs.factor, attention_dropout=configs.dropout,
        #                                   output_attention=False),
        #                     configs.d_model, configs.n_heads),
        #                 AttentionLayer(
        #                     FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                                   output_attention=False),
        #                     configs.d_model, configs.n_heads),
        #                 configs.d_model,
        #                 configs.d_ff,
        #                 dropout=configs.dropout,
        #                 activation=configs.activation,
        #             )
        #             for l in range(configs.d_layers)
        #         ],
        #         norm_layer=torch.nn.LayerNorm(configs.d_model),
        #         projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        #     )
        # self.rev = nn.Linear(9,configs.dec_in)
        # self.merg = nn.Linear(10,self.dmerging,bias=True)
        # self.merg2 = nn.Linear(4,configs.dec_in,bias=True)
        # self.prok = nn.Linear(4,configs.d_model,bias=True)
        # self.proq = nn.Linear(configs.dec_in, self.dmerging, bias=True)
        self.grl = nn.LSTM(self.pred_len,hidden_size=configs.seq_len)
        self.des = nn.Sequential()
        self.des.add_module('Linear',nn.Linear(configs.d_ff,self.pred_len))
        self.des.add_module('gelu',nn.Sigmoid())
        # self.proo = nn.Linear()
        self.proj = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self.prov = nn.Linear(configs.dec_in, configs.d_model,bias=True)
        self.Linear = nn.Sequential()
        self.Linear.add_module('Linear',nn.Linear(configs.seq_len, self.pred_len))
        # self.Linear.add_module('GELU', nn.GELU())
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin]*configs.dec_in),requires_grad=True)
        self.revin_layer = RevIN(configs.dec_in)
        self.updimlow = nn.Linear(4,configs.enc_in,bias=True)
        self.updimhigh = nn.Linear(4,configs.enc_in,bias=True)
        self.updimforcast = nn.Linear(3,configs.enc_in,bias=True)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #vision1

        # x_enc = self.revin_layer(x_enc, 'norm')
        # enc_in = x_enc.permute(0, 2, 1)
        # enc_out = self.encoder(enc_in)
        # dec_out = self.decoder(enc_out)
        # dec_out = dec_out + enc_out
        # dec_out2 = self.decoder(enc_in)
        # dec_out = dec_out * dec_out2
        # dec_out = self.proj(dec_out).permute(0, 2, 1)
        # linear_out = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        # dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out, 'denorm')

        #vision2  3不行，改为999
        # x_rev = self.rev(x_dec)
        # x_dec = self.revin_layer(x_rev, 'norm')
        # x_low = x_enc[:, :, [4,5,6,7,]]
        # x_high = x_enc[:,:,[0,1,2,3]]
        # x_low = self.merg(x_low)
        # enc_out = self.prok(x_high).permute(0, 2, 1)
        # emb_in = self.merg2(x_high)
        # dec_in = self.dec_embedding(emb_in, x_mark_enc)
        # dec_out = self.decoder(enc_out, dec_in)
        # linear_out = self.Linear(x_low.permute(0, 2, 1)).permute(0, 2, 1)
        # # dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out, 'denorm')
        # dec_out = dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out

        #vision3
        x_enc = self.revin_layer(x_enc, 'norm')
        x_enclow = x_enc[:, :, [4, 5, 6, 7]]
        x_enchigh = x_enc[:, :, [0, 1, 2, 3]]
        x_encforcast = x_enc[:, :, [8, 9, 10]]
        x_enclow = self.updimlow(x_enclow)
        x_enchigh = self.updimhigh(x_enchigh).permute(0, 2, 1)
        x_encforcast = self.updimforcast(x_encforcast).permute(0, 2, 1)
        enc_out = self.encoder(x_enchigh)
        dec_out = self.decoder(enc_out)
        dec_out = dec_out + enc_out
        dec_out2 = self.decoder(x_encforcast)
        dec_out = dec_out * dec_out2
        dec_out = self.proj(dec_out).permute(0, 2, 1)
        linear_out = self.Linear(x_enclow.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :] + self.w_dec * linear_out, 'denorm')


        if self.output_attention:

            return dec_out[:, -self.pred_len:, :] #,attns
        else:
            return dec_out