import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from transformers import BertModel, BertTokenizer
class BiAttention(nn.Module):
    # def __init__(self, input_dim, memory_dim, hid_dim, dropout):
    def __init__(self, hid_dim):
        super(BiAttention, self).__init__()
        # self.dropout = dropout
        self.input_linear_1 = nn.Linear(hid_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(hid_dim, 1, bias=False)

        self.input_linear_2 = nn.Linear(hid_dim, hid_dim, bias=True)
        self.memory_linear_2 = nn.Linear(hid_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(hid_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch.bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [f1, f2]^T [w1, w2] + <f1 * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot  # N x Ld x Lm
        # N * Ld * Lm
        att = att - 1e30 * (1 - mask[:, None])

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)
        # print('input',input.shape)
        # print('memo',memory.shape)
        # print('att',att.shape)
        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)
        # print('weight_two',weight_two.shape)
        # print('output_two',output_two.shape)
        # print(torch.cat([input, output_one, input*output_one, output_two*output_one], dim=1).shape)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=1)
        # return output_one