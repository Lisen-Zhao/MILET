import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead


from math import ceil

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index


    def forward(self, inputs):
        return inputs[self.item_index]
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = configs.output_attention
        self.lstm = nn.Sequential(
            nn.LSTM(configs.seq_len, hidden_size=configs.pred_len, batch_first=True),
        SelectItem(0),)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                    enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        dec_out = self.lstm(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        if self.output_attention:

            return dec_out[:, -self.pred_len:, :] #,attns
        else:
            return dec_out