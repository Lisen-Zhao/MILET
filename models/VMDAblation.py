import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.Crossformer_EncDec import Encoder, Decoder, DecoderLayer, DecoderLayernoseg
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
        # self.dmerging = 3
        self.seg_num = 1
        self.in_seg_num = configs.enc_in
        configs.d_model = configs.seq_len
        # self.win_size = 1

        self.lstm = nn.Sequential(
            TwoStageAttentionLayernoseg(configs, configs.dec_in, configs.factor, configs.d_model,
                                        configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),
            nn.LSTM(configs.seq_len, hidden_size=configs.seq_len, batch_first=True),
            SelectItem(0),
        TwoStageAttentionLayernoseg(configs, configs.dec_in, configs.factor, configs.d_model,
                                                                 configs.n_heads, configs.d_ff, configs.dropout),
                                     nn.LSTM(configs.seq_len, hidden_size=self.pred_len, batch_first=True),
                                     SelectItem(0),
                                     )
        self.lstm1 = nn.Sequential(
            TwoStageAttentionLayernoseg(configs, self.seg_num, configs.factor, configs.d_model,
                                       configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),
            nn.LSTM(configs.seq_len,hidden_size=configs.seq_len,batch_first=True),
            SelectItem(0),
            TwoStageAttentionLayernoseg(configs, self.seg_num, configs.factor, configs.d_model,
                                        configs.n_heads, configs.d_ff, configs.dropout),
            nn.LSTM(configs.seq_len, hidden_size=self.pred_len, batch_first=True),
            SelectItem(0),
         )

        self.lstm2 = nn.Sequential(
            TwoStageAttentionLayernoseg(configs, self.seg_num, configs.factor, configs.d_model,
                                        configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),
            nn.LSTM(configs.seq_len,hidden_size=self.pred_len,batch_first=True),
                                   SelectItem(0),
                                   )
        self.lstm3 = nn.Sequential(
            nn.LayerNorm(configs.seq_len),

            nn.LSTM(configs.seq_len,hidden_size=self.pred_len,batch_first=True),
                                   SelectItem(0),
                                   )
        self.lstm_other = nn.Sequential(
            TwoStageAttentionLayernoseg(configs, configs.dec_in - 4, configs.factor, configs.d_model,
                                        configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),
        )
        self.decoder_other = nn.Sequential(TwoStageAttentionLayernoseg(configs, configs.dec_in - 4, configs.factor, configs.d_model,
                                                                 configs.n_heads, configs.d_ff, configs.dropout),
                                     )

        self.padding = nn.Linear(1,configs.dec_in)
        self.outlinear = nn.Linear(configs.seq_len,self.pred_len)
        self.active = nn.GELU()
        self.norm = nn.LayerNorm(self.pred_len)
        self.comb = nn.Linear(configs.enc_in-1,1)

        self.revin_layer = RevIN(configs.dec_in)

        self.dropout = nn.Dropout(configs.dropout)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        #vision4
        x_enc = self.revin_layer(x_enc, 'norm').permute(0, 2, 1)
        # x_enc = x_enc.permute(0, 2, 1)
        net_out = torch.zeros([x_enc.size(0), x_enc.size(1), self.pred_len],
                    dtype=x_enc.dtype).to(x_enc.device)
        target =0
        # mfeat = self.lstm(x_enc)
        for i in range(x_enc.size(1)):
            if i == 0:
               # x_enc = torch.unsqueeze(x_enc[:,i,:],1)
               # x_enc = x_enc[:,i,:].unsqueeze(1)
                out = self.lstm1(x_enc[:, i, :].unsqueeze(1))


                # out = self.decoder(out)

                # out = self.outlinear[i](out)
                net_out[:, i, :] = out.squeeze()
                # target += net_out[:,i,:]
            elif i == 1:
                # x_enc = x_enc[:, i, :].unsqueeze(1)
                out = self.lstm2(x_enc[:, i, :].unsqueeze(1))
                # out = self.outlinear[i](out)
                net_out[:, i, :] = out.squeeze()
                # target += net_out[:, i, :]
            elif i == 2:
                out = self.lstm3(x_enc[:, i, :].unsqueeze(1))
                # out = self.outlinear[i](out)
                net_out[:, i, :] = out.squeeze()
                # target += net_out[:, i, :]


            else:
                other_ft = self.lstm_other(x_enc[:,3:-1,:])
                other_ft = self.decoder_other(other_ft)
                net_out[:,3:-1,:] = other_ft[:,:,-self.pred_len:]
                combine = net_out[:,:-1,:].permute(0, 2, 1)
                combine =self.comb(combine).permute(0, 2, 1).squeeze()
                out = self.active(self.outlinear(x_enc[:,i,:]))
                net_out[:,-1,:] = out+ self.dropout(combine)
                # net_out[:, -1, :] = out
                # net_out = net_out+self.dropout(mfeat)


        dec_out = net_out.permute(0, 2, 1)
        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :], 'denorm')
        # dec_out = dec_out[:, -self.pred_len:, :]
        if self.output_attention:

            return dec_out[:, -self.pred_len:, :] #,attns
        else:
            return dec_out