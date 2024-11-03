
import torch.nn as nn
from torch.autograd import Variable
import torch
from transformers import AutoModel
from torchcrf import CRF
from model.Multi_head_Attention import Multi_head_Attention
from config import Config
config = Config()
class ERNIE_GRU_CRF(nn.Module):
    """
    ernie_gru_crf model
    """
    def __init__(self, ernie_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False,use_llm=None):
        super(ERNIE_GRU_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # 加载ERNIE
        self.word_embeds = AutoModel.from_pretrained(ernie_config)
        # 使用GRU替换原来的LSTM
        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=rnn_layers, bidirectional=True,
                          dropout=dropout_ratio, batch_first=True)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(num_tags=tagset_size, batch_first=True)
        self.liner = nn.Linear(hidden_dim*2, tagset_size)
        self.tagset_size = tagset_size
        self.Multi_attention = Multi_head_Attention(d_model = self.hidden_dim * 2,n_head =config.num_heads )

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)) ## 改

        """
        修改了__init__函数和forward函数中的相关部分。
        同时，rand_init_hidden函数也做了调整，
        因为GRU的隐藏状态只有一个张量，而不是LSTM的两个（即隐藏状态和细胞状态）。
        """

    def forward(self, sentence, LLM_embedding,attention_mask=None,):
        '''
        args:
            sentence (batch_size, word_seq_len) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf input (batch_size, word_seq_len, tag_size), hidden
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds = self.word_embeds(sentence, attention_mask=attention_mask)
        if config.use_llm_embedding:
            embeds = embeds.last_hidden_state  ##改===============================
            LLM = LLM_embedding.unsqueeze(1).repeat(1, 100, 2)  ## LLM_embedding[50,384]  ##改===============================
            embeds = embeds + LLM ##改===============================
            hidden = self.rand_init_hidden(batch_size)
            if embeds[0].is_cuda:
                hidden = hidden.cuda()
            gru_out, hidden = self.gru(embeds, hidden)
            if config.use_attention:
                gru_out = self.Multi_attention( gru_out, gru_out, gru_out)
            gru_out = gru_out.contiguous().view(-1, self.hidden_dim*2)
            d_gru_out = self.dropout1(gru_out)
            l_out = self.liner(d_gru_out)
            gru_feats = l_out.contiguous().view(batch_size, seq_length, -1)
            return gru_feats
        else:
            hidden = self.rand_init_hidden(batch_size)
            if embeds[0].is_cuda:
                hidden = hidden.cuda()   ## 改
            gru_out, hidden = self.gru(embeds[0], hidden)
            if config.use_attention:
                gru_out = self.Multi_attention( gru_out, gru_out, gru_out)
            gru_out = gru_out.contiguous().view(-1, self.hidden_dim*2)
            d_gru_out = self.dropout1(gru_out)
            l_out = self.liner(d_gru_out)
            gru_feats = l_out.contiguous().view(batch_size, seq_length, -1)
            return gru_feats

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
        mask: size=(batch_size, seq_len)
        tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = -self.crf(feats, tags, mask)  # 计算损失
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value