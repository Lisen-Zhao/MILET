import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import OverallTwoStageAttentionLayer
from layers.RevIN import RevIN

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
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.seg_num = 1
        self.in_seg_num = configs.enc_in
        configs.d_model = configs.seq_len


        self.OTSA_LSTM_DOUBLE = nn.Sequential(
            OverallTwoStageAttentionLayer(configs, configs.dec_in, configs.factor, configs.d_model,
                                        configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),
            nn.LSTM(configs.seq_len, hidden_size=configs.seq_len, batch_first=True),
            SelectItem(0),
            OverallTwoStageAttentionLayer(configs, configs.dec_in, configs.factor, configs.d_model,
                                       configs.n_heads, configs.d_ff, configs.dropout),
            nn.LSTM(configs.seq_len, hidden_size=self.pred_len, batch_first=True),
            SelectItem(0),
                                     )
        self.OTSA_LSTM_1 = nn.Sequential(
            OverallTwoStageAttentionLayer(configs, self.seg_num, configs.factor, configs.d_model,
                                       configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),
            nn.LSTM(configs.seq_len,hidden_size=self.pred_len,batch_first=True),
            SelectItem(0),
         )
        self.OTSA_LSTM_2 = nn.Sequential(
            OverallTwoStageAttentionLayer(configs, self.seg_num, configs.factor, configs.d_model,
                                        configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),
            nn.LSTM(configs.seq_len,hidden_size=self.pred_len,batch_first=True),
                                   SelectItem(0),
                                   )
        self.LSTM = nn.Sequential(
            nn.LayerNorm(configs.seq_len),
            nn.LSTM(configs.seq_len,hidden_size=self.pred_len,batch_first=True),
                                   SelectItem(0),
                                   )
        self.OTSA_LSTM_3 = nn.Sequential(
            OverallTwoStageAttentionLayer(configs, configs.dec_in - 4, configs.factor, configs.d_model,
                                        configs.n_heads, configs.d_ff, configs.dropout),
            nn.LayerNorm(configs.seq_len),

        )
        self.active = nn.GELU()
        self.comb = nn.Linear(configs.enc_in-1,1)
        self.outlinear = nn.Linear(configs.seq_len, self.pred_len)
        self.revin_layer = RevIN(configs.dec_in)
        self.dropout = nn.Dropout(configs.dropout)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = self.revin_layer(x_enc, 'norm').permute(0, 2, 1)
        net_out = torch.zeros([x_enc.size(0), x_enc.size(1), self.pred_len],
                    dtype=x_enc.dtype).to(x_enc.device)
        mfeat = self.OTSA_LSTM_DOUBLE(x_enc)
        for i in range(x_enc.size(1)):
            """
            Single-dimensional three-modal part
            """
            if i == 0:
                out = self.OTSA_LSTM_1(x_enc[:, i, :].unsqueeze(1))
                net_out[:, i, :] = out.squeeze()
            if i == 1:
                out = self.OTSA_LSTM_2(x_enc[:, i, :].unsqueeze(1))
                net_out[:, i, :] = out.squeeze()
            if i == 2:
                out = self.LSTM(x_enc[:, i, :].unsqueeze(1))
                net_out[:, i, :] = out.squeeze()
                """
                Remaining features
                """
                other_ft = self.OTSA_LSTM_3(x_enc[:, 3:-1, :])
                net_out[:, 3:-1, :] = other_ft[:, :, -self.pred_len:]
                """
                Merge the above output
                """
                combine = net_out[:, :-1, :].permute(0, 2, 1)
                combine = self.comb(combine).permute(0, 2, 1).squeeze()
            if i >2 :
                """
                Linear part
                """
                out = self.active(self.outlinear(x_enc[:,i,:]))
                net_out[:,-1,:] = out+ self.dropout(combine)
        """
        Combined multidimensional
        """
        net_out = net_out+self.dropout(mfeat)
        dec_out = net_out.permute(0, 2, 1)
        """
        output
        """
        dec_out = self.revin_layer(dec_out[:, -self.pred_len:, :], 'denorm')
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :] #,attns
        else:
            return dec_out