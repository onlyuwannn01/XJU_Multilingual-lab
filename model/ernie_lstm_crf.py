# coding=utf-8
import torch.nn as nn
from torch.autograd import Variable
import torch
from transformers import AutoModel
from torchcrf import CRF
from model.Multi_head_Attention import Multi_head_Attention
from config import Config
config = Config()
from model.attention import Attention
"""
使用的crf是pytorch-crf. 安装方式： pip install pytorch-crf. pip install transformers pip install fire pip install seqeval
PyTorch 1.10
pip3 install torch torchvision torchaudio -i https://pypi.mirrors.ustc.edu.cn/simple/
"""



class ERNIE_LSTM_CRF(nn.Module):
    """
    ernie_lstm_crf model
    """
    def __init__(self, ernie_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False,use_llm=None):
        super(ERNIE_LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim ## 768
        self.hidden_dim = hidden_dim ## 512
        #加载ERNIE
        self.word_embeds = AutoModel.from_pretrained(ernie_config)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, ##双向0
                            dropout=dropout_ratio, batch_first=True)

        self.rnn_layers = rnn_layers ##1层
        self.dropout1 = nn.Dropout(p=dropout1) ##随机遗忘
        self.crf = CRF(num_tags=tagset_size, batch_first=True) ##CRF
        self.liner = nn.Linear(hidden_dim*2, tagset_size)  ##    ->10
        self.tagset_size = tagset_size
        #self.attention = Attention(embed_dim=self.tagset_size, hidden_dim=128, out_dim=self.tagset_size)
        self.Liner_3 = nn.Linear(self.tagset_size * 2, self.tagset_size)
        self.multi_attention = Multi_head_Attention(d_model = self.hidden_dim *2, n_head = 8)
    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim)), \
               Variable(torch.randn( 2*self.rnn_layers, batch_size, self.hidden_dim))


    def forward(self, sentence, LLM_embedding,attention_mask=None,):
        '''
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf input (batch_size, word_seq_len, tag_size), hidden
        '''
        seq_length = sentence.size(1) ## 100
        batch_size = sentence.size(0) ## 50
        embeds = self.word_embeds(sentence, attention_mask=attention_mask) ## 50 100 768
        if config.use_llm_embedding:

            embeds = embeds.last_hidden_state  ##改===============================
            LLM = LLM_embedding.unsqueeze(1).repeat(1, 100, 2)  ## LLM_embedding[50,384]  ##改===============================
            embeds = embeds + LLM ##改===============================
            hidden = self.rand_init_hidden(batch_size) ## 2 50 512
            if embeds[0].is_cuda:
                hidden = tuple(i.cuda() for i in hidden)
            ## 改！！！！！！！！！
            lstm_out, hidden = self.lstm(embeds, hidden) ## 50 100 1024
            if config.use_attention:
                lstm_out = self.multi_attention( lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2) ## 5000 1024
            d_lstm_out = self.dropout1(lstm_out) ## 5000 1024
            l_out = self.liner(d_lstm_out) ## 5000 10
            lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
            #lstm_feats, _ = self.attention(lstm_feats, lstm_feats)
            #lstm_feats = self.Liner_3(lstm_feats)
            return lstm_feats
        else:
            hidden = self.rand_init_hidden(batch_size)
            if embeds[0].is_cuda:
                hidden = tuple(i.cuda() for i in hidden)

            lstm_out, hidden = self.lstm(embeds[0], hidden)  ## 50 100 1024
            if config.use_attention:
                lstm_out = self.multi_attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)  ## 5000 1024
            d_lstm_out = self.dropout1(lstm_out)  ## 5000 1024
            l_out = self.liner(d_lstm_out)  ## 5000 10
            lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)

            return lstm_feats

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = -self.crf(feats, tags, mask) # 计算损失
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value






