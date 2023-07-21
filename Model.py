# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from biattention import BiAttention
import torch

class BERTModel(nn.Module):
    def __init__(self, args,):
        super(BERTModel, self).__init__()

        hidden_size = args.hidden_size
        self.biatt_a = BiAttention(hidden_size)
        self.biatt_o = BiAttention(hidden_size)
        self.biatt_ao = BiAttention(hidden_size)
        self.biatt_oa = BiAttention(hidden_size)
        # BERT模型
        if args.bert_model_type == 'bert-base-uncased':
            self._bert = BertModel.from_pretrained(args.bert_model_type)
            self._tokenizer = BertTokenizer.from_pretrained(args.bert_model_type)
            print('Bertbase model loaded')

        else:
            raise KeyError('Config.args.bert_model_type should be bert-based-uncased. ')
        self.classifier_a_reshape = nn.Linear(1940,117)
        self.classifier_b_reshape = nn.Linear(1940,117)

        self.classifier_a_start = nn.Linear(hidden_size, 2)
        self.classifier_a_end = nn.Linear(hidden_size, 2)
        self.classifier_ao_start = nn.Linear(hidden_size, 2)
        self.classifier_ao_end = nn.Linear(hidden_size, 2)
        self.classifier_o_start = nn.Linear(hidden_size, 2)
        self.classifier_o_end = nn.Linear(hidden_size, 2)
        self.classifier_oa_start = nn.Linear(hidden_size, 2)
        self.classifier_oa_end = nn.Linear(hidden_size, 2)
        self.classifier_sentiment = nn.Linear(hidden_size, 3)
        # self._classifier_category = nn.Linear(hidden_size, 121) #laptop 121 rest 13
        self._classifier_category_general = nn.Linear(hidden_size,23)
        self._classifier_attribute = nn.Linear(hidden_size,9)
    def forward(self, query_tensor, query_mask, query_seg, context_ids, step):
        c_emb = self._bert(context_ids, attention_mask=None)[0]
        hidden_states = self._bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
        sep_token = torch.tensor([self._tokenizer.sep_token_id]).cuda()
        sep_positions = (query_seg == sep_token).nonzero()
        if sep_positions.numel() == 0:
            # only one sentence in the input
            first_sep_position = query_tensor.size(1)  # use the length of the input as the separator position
        else:
            first_sep_position = sep_positions[:, 1].min().item()
        sentence_embedding = hidden_states[:, :first_sep_position+1, :]
        # print('hidd',hidden_states.shape)
        if step == 'A':
            hidden_states = self.biatt_a(c_emb,hidden_states, query_mask)
            hidden_states =self.classifier_a_reshape(hidden_states.permute(0, 2, 1))
            hidden_states = hidden_states.permute(0, 2, 1)
            predict_start = self.classifier_a_start(hidden_states)
            predict_end = self.classifier_a_end(hidden_states)
            return predict_start, predict_end
        elif step == 'O':
            hidden_states = self.biatt_o(c_emb,hidden_states, query_mask)
            hidden_states =self.classifier_a_reshape(hidden_states.permute(0, 2, 1))
            hidden_states = hidden_states.permute(0, 2, 1)
            predict_start = self.classifier_o_start(hidden_states)
            predict_end = self.classifier_o_end(hidden_states)
            return predict_start, predict_end
        elif step == 'AO':
            c_emb_extend = []
            for c in c_emb:
                for i in range(int(hidden_states.shape[0]/c_emb.shape[0])):
                    c_emb_extend.append(c)
            c_emb_extend = torch.tensor([item.cpu().detach().numpy() for item in c_emb_extend]).cuda()

            hidden_states = self.biatt_ao(c_emb_extend,sentence_embedding, query_mask)
            hidden_states =self.classifier_b_reshape(hidden_states.permute(0, 2, 1))
            hidden_states = hidden_states.permute(0, 2, 1)
            predict_start = self.classifier_ao_start(hidden_states)
            predict_end = self.classifier_ao_end(hidden_states)
            # print('predict_start', predict_start.shape)
            return predict_start, predict_end
        elif step == 'OA':
            c_emb_extend = []
            for c in c_emb:
                for i in range(int(hidden_states.shape[0]/c_emb.shape[0])):
                    c_emb_extend.append(c)
            c_emb_extend = torch.tensor([item.cpu().detach().numpy() for item in c_emb_extend]).cuda()
            hidden_states = self.biatt_oa(c_emb_extend,sentence_embedding, query_mask)
            hidden_states =self.classifier_b_reshape(hidden_states.permute(0, 2, 1))
            hidden_states = hidden_states.permute(0, 2, 1)
            predict_start = self.classifier_oa_start(hidden_states)
            predict_end = self.classifier_oa_end(hidden_states)
            return predict_start, predict_end
        elif step == 'S':
            sentiment_hidden_states = hidden_states[:, 0, :]
            # print(sentiment_hidden_states)
            sentiment_scores = self.classifier_sentiment(sentiment_hidden_states)
            return sentiment_scores
        elif step == 'C':
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_category(cls_hidden_states)
            return cls_hidden_scores
        elif step == 'CG':
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_category_general(cls_hidden_states)
            return cls_hidden_scores
        elif step == 'CA':
            cls_hidden_states = hidden_states[:, 0, :]
            cls_hidden_scores = self._classifier_attribute(cls_hidden_states)
            return cls_hidden_scores
