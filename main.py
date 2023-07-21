# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import argparse
import Data
import Model
import utils
import torch
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
import os
from torch.utils.data import Dataset
import random
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
scaler = torch.cuda.amp.GradScaler()
torch.cuda.init()

class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self._forward_asp_query = pre_data['_forward_asp_query']
        self._forward_opi_query = pre_data['_forward_opi_query']
        self._forward_asp_answer_start = pre_data['_forward_asp_answer_start']
        self._forward_asp_answer_end = pre_data['_forward_asp_answer_end']
        self._forward_opi_answer_start = pre_data['_forward_opi_answer_start']
        self._forward_opi_answer_end = pre_data['_forward_opi_answer_end']
        self._forward_asp_query_mask = pre_data['_forward_asp_query_mask']
        self._forward_opi_query_mask = pre_data['_forward_opi_query_mask']
        self._forward_asp_query_seg = pre_data['_forward_asp_query_seg']
        self._forward_opi_query_seg = pre_data['_forward_opi_query_seg']

        self._backward_asp_query = pre_data['_backward_asp_query']
        self._backward_opi_query = pre_data['_backward_opi_query']
        self._backward_asp_answer_start = pre_data['_backward_asp_answer_start']
        self._backward_asp_answer_end = pre_data['_backward_asp_answer_end']
        self._backward_opi_answer_start = pre_data['_backward_opi_answer_start']
        self._backward_opi_answer_end = pre_data['_backward_opi_answer_end']
        self._backward_asp_query_mask = pre_data[
            '_backward_asp_query_mask']
        self._backward_opi_query_mask = pre_data[
            '_backward_opi_query_mask']
        self._backward_asp_query_seg = pre_data['_backward_asp_query_seg']
        self._backward_opi_query_seg = pre_data['_backward_opi_query_seg']

        self._sentiment_query = pre_data['_sentiment_query']
        self._sentiment_answer = pre_data['_sentiment_answer']
        self._sentiment_query_mask = pre_data['_sentiment_query_mask']
        self._sentiment_query_seg = pre_data['_sentiment_query_seg']

        # self._cate_query = pre_data['_cate_query']
        # self._cate_answer = pre_data['_cate_answer']
        # self._cate_query_mask = pre_data['_cate_query_mask']
        # self._cate_query_seg = pre_data['_cate_query_seg']

        self._category_query = pre_data['_category_query']
        self._category_answer = pre_data['_cate_answer']
        self._category_query_mask = pre_data['_category_query_mask']
        self._category_query_seg = pre_data['_category_query_seg']

        self._attribute_query = pre_data['_attribute_query']
        self._attribute_answer = pre_data['_attribute_answer']
        self._attribute_query_mask = pre_data['_attribute_query_mask']
        self._attribute_query_seg = pre_data['_attribute_query_seg']

        self._aspect_num = pre_data['_aspect_num']
        self._opinion_num = pre_data['_opinion_num']


def test(model, t, batch_generator, standard, beta, logger):
    model.eval()

    triplet_target_num = 0
    category_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_pol_target_num = 0
    # asp_cate_target_num = 0
    asp_category_target_num = 0
    asp_attribute_target_num = 0

    triplet_predict_num = 0
    category_predict_num = 0
    asp_predict_num = 0
    opi_predict_num = 0
    asp_opi_predict_num = 0
    asp_pol_predict_num = 0
    # asp_cate_predict_num = 0
    asp_category_predict_num = 0
    asp_attribute_predict_num =0

    triplet_match_num = 0
    category_match_num =0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_pol_match_num = 0
    # asp_cate_match_num = 0
    asp_category_match_num =0
    asp_attribute_match_num = 0


    for batch_index, batch_dict in enumerate(batch_generator):

        triplets_target = standard[batch_index]['triplet']
        # print('triplets target:', triplets_target)
        categorys_target = standard[batch_index]['category_pair']
        asp_target = standard[batch_index]['asp_target']
        opi_target = standard[batch_index]['opi_target']
        asp_opi_target = standard[batch_index]['asp_opi_target']
        asp_pol_target = standard[batch_index]['asp_pol_target']
        # asp_cate_target = standard[batch_index]['asp_cate_target']
        asp_category_target =standard[batch_index]['asp_category_target']
        asp_attribute_target =standard[batch_index]['asp_attribute_target']

        # 预测三元组
        triplets_predict = []
        categorys_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_pol_predict = []
        # asp_cate_predict = []
        asp_category_predict = []
        asp_attribute_predict = []

        forward_pair_list = []
        forward_pair_prob = []
        forward_pair_ind_list = []

        backward_pair_list = []
        backward_pair_prob = []
        backward_pair_ind_list = []

        final_asp_list = []
        final_opi_list = []
        final_asp_ind_list = []
        final_opi_ind_list = []
        # forward q_1
        passenge_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()
        passenge = batch_dict['forward_asp_query'][0][passenge_index].squeeze(1)
        # print('passenge',passenge)


        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'],batch_dict['context'], 'A')
        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []
        for i in range(f_asp_start_ind.size(0)):
            if batch_dict['forward_asp_answer_start'][0, i] != -1:
                if f_asp_start_ind[i].item() == 1:
                    f_asp_start_index_temp.append(i)
                    f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
                if f_asp_end_ind[i].item() == 1:
                    f_asp_end_index_temp.append(i)
                    f_asp_end_prob_temp.append(f_asp_end_prob[i].item())


        f_asp_start_index, f_asp_end_index, f_asp_prob = utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp)

        for i in range(len(f_asp_start_index)):
            opinion_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(t.convert_tokens_to_ids('?'))
            opinion_query.append(t.convert_tokens_to_ids('[SEP]'))
            
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query)
            opinion_query = torch.tensor(opinion_query).long().cuda()
            opinion_query = torch.cat([opinion_query, passenge], -1).unsqueeze(0)
            opinion_query_seg += [1] * passenge.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().cuda().unsqueeze(0)
            opinion_query_seg = torch.tensor(opinion_query_seg).long().cuda().unsqueeze(0)

            N = opinion_query.size(1)
            dim_to_pad = 1
            pad_length = 117 - N

            # 执行填充操作
            opinion_query = torch.nn.functional.pad(opinion_query, (0, pad_length), mode='constant', value=0)
            opinion_query_mask = torch.nn.functional.pad(opinion_query_mask, (0, pad_length), mode='constant', value=0)
            opinion_query_seg = torch.nn.functional.pad(opinion_query_seg, (0, pad_length), mode='constant', value=1)
            # print('opinion query seg', opinion_query_seg.shape)
            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg,batch_dict['context'], 'AO')
            # print('opinion query seg', opinion_query_seg.shape)
            # print('opinion_query',opinion_query.shape)
            # print('opinion_query_mask',opinion_query_mask.shape)

            f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
            f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
            f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
            f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

            f_opi_start_prob_temp = []
            f_opi_end_prob_temp = []
            f_opi_start_index_temp = []
            f_opi_end_index_temp = []
            for k in range(f_opi_start_ind.size(0)):
                if opinion_query_seg[0, k] == 1:
                    if f_opi_start_ind[k].item() == 1:
                        f_opi_start_index_temp.append(k)
                        f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                    if f_opi_end_ind[k].item() == 1:
                        f_opi_end_index_temp.append(k)
                        f_opi_end_prob_temp.append(f_opi_end_prob[k].item())


            f_opi_start_index, f_opi_end_index, f_opi_prob = utils.filter_unpaired(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp)


            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[i]-5, f_asp_end_index[i]-5]
                opi_ind = [f_opi_start_index[idx]-f_opi_length, f_opi_end_index[idx]-f_opi_length]
                #####probabilty 
                temp_prob = math.sqrt(f_asp_prob[i] * f_opi_prob[idx])
                if asp_ind + opi_ind not in forward_pair_list:
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)

        # backward q_1
        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'],
                                                     batch_dict['context'],
                                                      'O')
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)


        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for i in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, i] != -1:
                if b_opi_start_ind[i].item() == 1:
                    b_opi_start_index_temp.append(i)
                    b_opi_start_prob_temp.append(b_opi_start_prob[i].item())
                if b_opi_end_ind[i].item() == 1:
                    b_opi_end_index_temp.append(i)
                    b_opi_end_prob_temp.append(b_opi_end_prob[i].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp)



        # backward q_2
        for i in range(len(b_opi_start_index)):
            aspect_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[i], b_opi_end_index[i] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(t.convert_tokens_to_ids('describe'))
            aspect_query.append(t.convert_tokens_to_ids('?'))
            aspect_query.append(t.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long().cuda()
            aspect_query = torch.cat([aspect_query, passenge], -1).unsqueeze(0)
            aspect_query_seg += [1] * passenge.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().cuda().unsqueeze(0)
            aspect_query_seg = torch.tensor(aspect_query_seg).long().cuda().unsqueeze(0)

            N = aspect_query.size(1)
            dim_to_pad = 1
            pad_length = 117 - N

            # 执行填充操作
            aspect_query = torch.nn.functional.pad(aspect_query, (0, pad_length), mode='constant', value=0)
            aspect_query_mask = torch.nn.functional.pad(aspect_query_mask, (0, pad_length), mode='constant', value=0)
            aspect_query_seg = torch.nn.functional.pad(aspect_query_seg, (0, pad_length), mode='constant', value=1)

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg,batch_dict['context'], 'OA')
            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[i], b_opi_end_index[i] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx]-b_asp_length, b_asp_end_index[idx]-b_asp_length]
                opi_ind = [b_opi_start_index[i]-5, b_opi_end_index[i]-5]
                # TODO
                temp_prob = math.sqrt(b_asp_prob[idx] * b_opi_prob[i])
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('erro')
                    exit(1)
        # filter triplet
        # forward
        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list:
                if forward_pair_list[idx][0] not in final_asp_list:
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else:
                    asp_index = final_asp_list.index(forward_pair_list[idx][0])
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
            else:
                if forward_pair_prob[idx] >= beta:
                    if forward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(forward_pair_list[idx][0])
                        final_opi_list.append([forward_pair_list[idx][1]])
                        final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(forward_pair_list[idx][0])
                        if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(forward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
         # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])

    # sentiment + category###########
        for idx in range(len(final_asp_list)):
            predict_opinion_num = len(final_opi_list[idx])
            sentiment_query = t.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What sentiment given the aspect'.split(' ')])
            sentiment_query+=final_asp_list[idx]
            sentiment_query += t.convert_tokens_to_ids([word.lower() for word in 'and the opinion'.split(' ')])
            # # 拼接所有的opinion
            for idy in range(predict_opinion_num):
                sentiment_query+=final_opi_list[idx][idy]
                if idy < predict_opinion_num - 1:
                    sentiment_query.append(t.convert_tokens_to_ids('/'))
            sentiment_query.append(t.convert_tokens_to_ids('?'))
            sentiment_query.append(t.convert_tokens_to_ids('[SEP]'))

            sentiment_query_seg = [0] * len(sentiment_query)
            sentiment_query = torch.tensor(sentiment_query).long().cuda()
            sentiment_query = torch.cat([sentiment_query, passenge], -1).unsqueeze(0)
            sentiment_query_seg += [1] * passenge.size(0)
            sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
            sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

            sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg,batch_dict['context'], 'S')
            sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()
#########################cate查询我加的#################################
            # cate_query = t.convert_tokens_to_ids([word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
            #      '[CLS] What cate does the aspect'.split(' ')])
            # cate_query+=final_asp_list[idx]
            # cate_query += t.convert_tokens_to_ids([word.lower() for word in 'belong to ?'.split(' ')])
            # cate_query.append(t.convert_tokens_to_ids('[SEP]'))
            # cate_query_seg = [0] * len(cate_query)
            # cate_query = torch.tensor(cate_query).long().cuda()
            # cate_query = torch.cat([cate_query, passenge], -1).unsqueeze(0)
            # cate_query_seg += [1] * passenge.size(0)
            # cate_query_mask = torch.ones(cate_query.size(1)).float().cuda().unsqueeze(0)
            # cate_query_seg = torch.tensor(cate_query_seg).long().cuda().unsqueeze(0)

            # cate_scores = model(cate_query, cate_query_mask, cate_query_seg, 'C')
            # cate_predicted = torch.argmax(cate_scores[0], dim=0).item()

#########################category查询我加的#################################
            category_query = t.convert_tokens_to_ids([word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What general category does the aspect'.split(' ')])
            category_query +=final_asp_list[idx]
            category_query += t.convert_tokens_to_ids([word.lower() for word in 'belong to ?'.split(' ')])
            category_query.append(t.convert_tokens_to_ids('[SEP]'))
            category_query_seg = [0] * len(category_query)
            category_query = torch.tensor(category_query).long().cuda()
            category_query = torch.cat([category_query, passenge], -1).unsqueeze(0)
            category_query_seg += [1] * passenge.size(0)
            category_query_mask = torch.ones(category_query.size(1)).float().cuda().unsqueeze(0)
            category_query_seg = torch.tensor(category_query_seg).long().cuda().unsqueeze(0)

            category_scores = model(category_query, category_query_mask, category_query_seg,batch_dict['context'], 'CG')
            category_predicted = torch.argmax(category_scores[0], dim=0).item()
            # print('category_predicted',category_predicted)

#########################attribute查询我加的#################################
            attribute_query = t.convert_tokens_to_ids([word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What attribute of the category'.split(' ')])
            # attribute_query += category_predicted
            attribute_query += t.convert_tokens_to_ids([word.lower() for word in 'does the aspect'.split(' ')])
            attribute_query+=final_asp_list[idx]
            attribute_query += t.convert_tokens_to_ids([word.lower() for word in 'belong to ?'.split(' ')])
            attribute_query.append(t.convert_tokens_to_ids('[SEP]'))
            attribute_query_seg = [0] * len(attribute_query)
            attribute_query = torch.tensor(attribute_query).long().cuda()
            attribute_query = torch.cat([attribute_query, passenge], -1).unsqueeze(0)
            attribute_query_seg += [1] * passenge.size(0)
            attribute_query_mask = torch.ones(attribute_query.size(1)).float().cuda().unsqueeze(0)
            attribute_query_seg = torch.tensor(attribute_query_seg).long().cuda().unsqueeze(0)

            attribute_scores = model(attribute_query, attribute_query_mask, attribute_query_seg,batch_dict['context'], 'CA')
            attribute_predicted = torch.argmax(attribute_scores[0], dim=0).item()

            # 每个opinion对应一个4元组
            for idy in range(predict_opinion_num):
                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])
                # triplet_predict = asp_f + opi_f + [sentiment_predicted] +[cate_predicted]
                category_predict = [category_predicted, attribute_predicted]
                triplet_predict = asp_f + opi_f + [sentiment_predicted] + category_predict
                # print('predicet:',triplet_predict)
                # print('categorypredict:', category_predict)
                triplets_predict.append(triplet_predict)
                categorys_predict.append(category_predict)
                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)
                if asp_f + [sentiment_predicted] not in asp_pol_predict:
                    asp_pol_predict.append(asp_f + [sentiment_predicted])
                # if asp_f + [cate_predicted] not in asp_cate_predict:
                #     asp_cate_predict.append(asp_f + [cate_predicted])
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)
                if category_predicted not in asp_category_predict:
                    asp_category_predict.append(category_predicted)
                if attribute_predicted not in asp_attribute_predict:
                    asp_attribute_predict.append(attribute_predicted)

        triplet_target_num += len(triplets_target)
        asp_target_num += len(asp_target)
        opi_target_num += len(opi_target)
        asp_opi_target_num += len(asp_opi_target)
        asp_pol_target_num += len(asp_pol_target)
        # asp_cate_target_num +=len(asp_cate_target)
        asp_category_target_num += len(asp_category_target)
        asp_attribute_target_num += len(asp_attribute_target)
        category_target_num += len(categorys_target)

        triplet_predict_num += len(triplets_predict)
        asp_predict_num += len(asp_predict)
        opi_predict_num += len(opi_predict)
        asp_opi_predict_num += len(asp_opi_predict)
        asp_pol_predict_num += len(asp_pol_predict)
        # asp_cate_predict_num += len(asp_cate_predict)
        asp_category_predict_num += len(asp_category_predict)
        asp_attribute_predict_num += len(asp_attribute_predict)
        category_predict_num += len(categorys_predict)

        for trip in triplets_target:
            # print('trip:',trip)
            for trip_ in triplets_predict:
                if trip_ == trip:
                    triplet_match_num += 1
        for trip in asp_target:
            for trip_ in asp_predict:
                if trip_ == trip:
                    asp_match_num += 1
        for trip in opi_target:
            for trip_ in opi_predict:
                if trip_ == trip:
                    opi_match_num += 1
        for trip in asp_opi_target:
            for trip_ in asp_opi_predict:
                if trip_ == trip:
                    asp_opi_match_num += 1
        for trip in asp_pol_target:
            for trip_ in asp_pol_predict:
                if trip_ == trip:
                    asp_pol_match_num += 1
        # for trip in asp_cate_target:
        #     for trip_ in asp_cate_predict:
        #         if trip_ == trip:
        #             asp_cate_match_num += 1 

        # for trip in asp_category_target:
        #     for trip_ in asp_category_predict:
        #         if trip_ == trip:
        #             asp_category_match_num += 1
        # for trip in asp_attribute_target:
        #     for trip_ in asp_attribute_predict:
        #         if trip_ == trip:
        #             asp_attribute_match_num += 1
        for trip in categorys_target:
            for trip_ in categorys_predict:
                if trip_ == trip:
                    category_match_num += 1         

    precision = float(triplet_match_num) / float(triplet_predict_num+1e-6)
    recall = float(triplet_match_num) / float(triplet_target_num+1e-6)
    f1 = 2 * precision * recall / (precision + recall+1e-6)
    logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))


    precision_aspect = float(asp_match_num) / float(asp_predict_num+1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num+1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect+1e-6)
    logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_opinion = float(opi_match_num) / float(opi_predict_num+1e-6)
    recall_opinion = float(opi_match_num) / float(opi_target_num+1e-6)
    f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion+1e-6)
    logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

    precision_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_predict_num+1e-6)
    recall_aspect_sentiment = float(asp_pol_match_num) / float(asp_pol_target_num+1e-6)
    f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
            precision_aspect_sentiment + recall_aspect_sentiment+1e-6)
    logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                              recall_aspect_sentiment,
                                                                              f1_aspect_sentiment))
                                                                    
    # precision_aspect_category = float(asp_cate_match_num) / float(asp_cate_predict_num+1e-6)
    # recall_aspect_category = float(asp_cate_match_num) / float(asp_cate_target_num+1e-6)
    # f1_aspect_category = 2 * precision_aspect_category * recall_aspect_category / (
    #         precision_aspect_category + recall_aspect_category+1e-6)
    # logger.info('Aspect-Category - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_category,
    #                                                                           recall_aspect_category,
    #                                                                           f1_aspect_category))                                                                    

    precision_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_predict_num+1e-6)
    recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num+1e-6)
    f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
            precision_aspect_opinion + recall_aspect_opinion+1e-6)
    logger.info(
        'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                    f1_aspect_opinion))
    # precision_aspect_category_general = float(asp_category_match_num) / float(asp_category_predict_num+1e-6)
    # recall_aspect_category_general = float(asp_category_match_num) / float(asp_category_target_num+1e-6)
    # f1_aspect_category_general = 2 * precision_aspect_category_general * recall_aspect_category_general / (
    #         precision_aspect_category_general + recall_aspect_category_general+1e-6)
    # logger.info('Aspect-Category-General - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_category_general,
    #                                                                           recall_aspect_category_general,
    #                                                                           f1_aspect_category_general))
    # precision_aspect_attribute = float(asp_attribute_match_num) / float(asp_attribute_predict_num+1e-6)
    # recall_aspect_attribute = float(asp_attribute_match_num) / float(asp_attribute_target_num+1e-6)
    # f1_aspect_attribute = 2 * precision_aspect_attribute * recall_aspect_attribute / (
    #         precision_aspect_attribute + recall_aspect_attribute+1e-6)
    # logger.info('Aspect-Category-attribute - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_attribute,
    #                                                                           recall_aspect_attribute,
    #                                                                           f1_aspect_attribute))
    precision_category_pair = float(category_match_num) / float(category_predict_num+1e-6)
    recall_category_pair = float(category_match_num) / float(category_target_num+1e-6)
    f1_category_pair = 2 * precision_category_pair * recall_category_pair / (
            precision_category_pair + recall_category_pair+1e-6)
    logger.info('Aspect-Category-pair - Precision: {}\tRecall: {}\tF1: {}'.format(precision_category_pair,
                                                                              recall_category_pair,
                                                                              f1_category_pair))                                                                                                                                                               
    return f1


def main(args, tokenize):
    args.log_path = args.log_path + args.data_name + '_' + args.model_name + '.log'
    data_path = args.data_path + args.data_name + '.pt'
    standard_data_path = args.data_path + args.data_name + '_standard.pt'

    # init logger
    logger = utils.get_logger(args.log_path)

    # load data
    logger.info('loading data......')
    total_data = torch.load(data_path)
    standard_data = torch.load(standard_data_path)
    
    train_data = total_data['train']
    dev_data = total_data['dev']
    test_data = total_data['test']

    dev_standard = standard_data['dev']
    test_standard = standard_data['test']


    # max_aspect_num = total_data[args.max_aspect_num]
    # init model
    logger.info('initial model......')
    model = Model.BERTModel(args)
    if args.ifgpu:
        model = model.cuda()

    # print args
    logger.info(args)

    if args.mode == 'test':
        logger.info('start testing......')
        test_dataset = Data.ReviewDataset(train_data, dev_data, test_data, 'test')
        # load checkpoint
        logger.info('loading checkpoint......')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['net'])
        model.eval()

        batch_generator_test = Data.generate_fi_batches(dataset=test_dataset, batch_size=1, shuffle=False,
                                                        ifgpu=args.ifgpu)
        # eval
        logger.info('evaluating......')
        f1 = test(model, tokenize, batch_generator_test, test_standard, args.beta, logger)


    elif args.mode == 'train':
        args.save_model_path = args.save_model_path + args.data_name + '_' + args.model_name + '.pth'
        train_dataset = Data.ReviewDataset(train_data, dev_data, test_data, 'train')
        dev_dataset = Data.ReviewDataset(train_data, dev_data, test_data, 'dev')
        test_dataset = Data.ReviewDataset(train_data, dev_data, test_data, 'test')
        batch_num_train = train_dataset.get_batch_num(args.batch_size)

        # optimizer
        logger.info('initial optimizer......')
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "_bert" in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if "_bert" not in n],
             'lr': args.learning_rate, 'weight_decay': 0.01}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.tuning_bert_rate, correct_bias=False)

        # load saved model, optimizer and epoch num
        if args.reload and os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info('Reload model and optimizer after training epoch {}'.format(checkpoint['epoch']))
        else:
            start_epoch = 1
            logger.info('New model and optimizer from epoch 0')

        # scheduler
        training_steps = args.epoch_num * batch_num_train
        warmup_steps = int(training_steps * args.warm_up)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)

        # training
        logger.info('begin training......')
        best_dev_f1 = 0.
        for epoch in range(start_epoch, args.epoch_num+1):
            model.train()
            model.zero_grad()

            batch_generator = Data.generate_fi_batches(dataset=train_dataset, batch_size=args.batch_size,
                                                       ifgpu=args.ifgpu)

            for batch_index, batch_dict in enumerate(batch_generator):

                optimizer.zero_grad()

                # q1_a
                f_aspect_start_scores, f_aspect_end_scores = model(batch_dict['forward_asp_query'],
                                                                   batch_dict['forward_asp_query_mask'],
                                                                   batch_dict['forward_asp_query_seg'],batch_dict['context'], 'A')
                f_asp_loss = utils.calculate_entity_loss(f_aspect_start_scores, f_aspect_end_scores,
                                                                    batch_dict['forward_asp_answer_start'],
                                                                    batch_dict['forward_asp_answer_end'])
                # q1_b
                b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                             batch_dict['backward_opi_query_mask'],
                                                             batch_dict['backward_opi_query_seg'],batch_dict['context'], 'O')
                b_opi_loss = utils.calculate_entity_loss(b_opi_start_scores, b_opi_end_scores,
                                                                    batch_dict['backward_opi_answer_start'],
                                                                    batch_dict['backward_opi_answer_end'])
                # q2_a
                f_opi_start_scores, f_opi_end_scores = model(
                    batch_dict['forward_opi_query'].view(-1, batch_dict['forward_opi_query'].size(-1)),
                    batch_dict['forward_opi_query_mask'].view(-1, batch_dict['forward_opi_query_mask'].size(-1)),
                    batch_dict['forward_opi_query_seg'].view(-1, batch_dict['forward_opi_query_seg'].size(-1)),batch_dict['context'],
                    'AO')
                f_opi_loss = utils.calculate_entity_loss(f_opi_start_scores, f_opi_end_scores,
                                                         batch_dict['forward_opi_answer_start'].view(-1, batch_dict['forward_opi_answer_start'].size(-1)),
                                                         batch_dict['forward_opi_answer_end'].view(-1, batch_dict['forward_opi_answer_end'].size(-1)))
                # q2_b
                b_asp_start_scores, b_asp_end_scores = model(
                    batch_dict['backward_asp_query'].view(-1, batch_dict['backward_asp_query'].size(-1)),
                    batch_dict['backward_asp_query_mask'].view(-1, batch_dict['backward_asp_query_mask'].size(-1)),
                    batch_dict['backward_asp_query_seg'].view(-1, batch_dict['backward_asp_query_seg'].size(-1)),batch_dict['context'],
                    'OA')
                b_asp_loss = utils.calculate_entity_loss(b_asp_start_scores, b_asp_end_scores,
                                                         batch_dict['backward_asp_answer_start'].view(-1, batch_dict['backward_asp_answer_start'].size(-1)),
                                                         batch_dict['backward_asp_answer_end'].view(-1, batch_dict['backward_asp_answer_end'].size(-1)))
                # q_3
                sentiment_scores = model(batch_dict['sentiment_query'].view(-1, batch_dict['sentiment_query'].size(-1)),
                                         batch_dict['sentiment_query_mask'].view(-1, batch_dict['sentiment_query_mask'].size(-1)),
                                         batch_dict['sentiment_query_seg'].view(-1, batch_dict['sentiment_query_seg'].size(-1)),batch_dict['context'],
                                         'S')
                sentiment_loss = utils.calculate_sentiment_loss(sentiment_scores, batch_dict['sentiment_answer'].view(-1))

                # # q_4
                # cate_scores = model(batch_dict['cate_query'].view(-1, batch_dict['cate_query'].size(-1)),
                #                          batch_dict['cate_query_mask'].view(-1, batch_dict['cate_query_mask'].size(-1)),
                #                          batch_dict['cate_query_seg'].view(-1, batch_dict['cate_query_seg'].size(-1)),
                #                          'C')
                # cate_loss = utils.calculate_cate_loss(cate_scores, batch_dict['cate_answer'].view(-1))

                # q_5
                category_scores = model(batch_dict['category_query'].view(-1, batch_dict['category_query'].size(-1)),
                                         batch_dict['category_query_mask'].view(-1, batch_dict['category_query_mask'].size(-1)),
                                         batch_dict['category_query_seg'].view(-1, batch_dict['category_query_seg'].size(-1)),batch_dict['context'],
                                         'CG')
                category_loss = utils.calculate_cate_loss(category_scores, batch_dict['category_answer'].view(-1))

                # q_6
                attribute_scores = model(batch_dict['attribute_query'].view(-1, batch_dict['attribute_query'].size(-1)),
                                         batch_dict['attribute_query_mask'].view(-1, batch_dict['attribute_query_mask'].size(-1)),
                                         batch_dict['attribute_query_seg'].view(-1, batch_dict['attribute_query_seg'].size(-1)),batch_dict['context'],
                                         'CA')
                attribute_loss = utils.calculate_cate_loss(attribute_scores, batch_dict['attribute_answer'].view(-1))

                # loss
                loss_sum = f_asp_loss + f_opi_loss + b_opi_loss + b_asp_loss + args.beta*(sentiment_loss +category_loss+ attribute_loss) 
                # scaler.scale(loss_sum).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss_sum.backward()
                optimizer.step()
                scheduler.step()

                # train logger
                if batch_index % 10 == 0:
                    logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                                'forward Loss:{};{}\t backward Loss:{};{}\t Sentiment Loss:{} \t Category_general loss:{} \t Category attribute loss:{}'.format(epoch, args.epoch_num, batch_index, batch_num_train,
                                       round(loss_sum.item(), 4),
                                       round(f_asp_loss.item(), 4), round(f_opi_loss.item(), 4),
                                       round(b_asp_loss.item(), 4), round(b_opi_loss.item(), 4),
                                       round(sentiment_loss.item(), 4),
                                       round(category_loss.item(), 4),
                                       round(attribute_loss.item(), 4)))
                # if batch_index % 10 == 0:
                #     logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                #                 'forward Loss:{};{}\t backward Loss:{};{}\t Sentiment Loss:{} ' .
                #                 format(epoch, args.epoch_num, batch_index, batch_num_train,
                #                        round(loss_sum.item(), 4),
                #                        round(f_asp_loss.item(), 4), round(f_opi_loss.item(), 4),
                #                        round(b_asp_loss.item(), 4), round(b_opi_loss.item(), 4),
                #                        round(sentiment_loss.item(), 4),
                #                        ))

            # validation
            batch_generator_dev = Data.generate_fi_batches(dataset=dev_dataset, batch_size=1, shuffle=False,
                                                           ifgpu=args.ifgpu)
            f1 = test(model, tokenize, batch_generator_dev, dev_standard, args.inference_beta, logger)
            # save model and optimizer
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                logger.info('Model saved after epoch {}'.format(epoch))
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, args.save_model_path)

            # test
            batch_generator_test = Data.generate_fi_batches(dataset=test_dataset, batch_size=1, shuffle=False,
                                                            ifgpu=args.ifgpu)
            f1 = test(model, tokenize, batch_generator_test, test_standard, args.inference_beta, logger)

    else:
        logger.info('Error mode!')
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bidirectional MRC-based sentiment triplet extraction')
    parser.add_argument('--data_path', type=str, default="./data/preprocess/")
    parser.add_argument('--log_path', type=str, default="./log/")
    parser.add_argument('--data_name', type=str, default='Laptop_ACOS', choices=["rest15", "rest16","Laptop_ACOS",'Restaurant_ACOS'])

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    # parser.add_argument('--max_len', type=str, default="max_len", choices=["max_len"])
    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="./model/Laptop_ACOS/modelFinal.model")
    parser.add_argument('--save_model_path', type=str, default="./model/")
    parser.add_argument('--model_name', type=str, default="13biatt")

    # model hyper-parameter
    parser.add_argument('--bert_model_type', type=str, default="bert-base-uncased")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.8)

    # training hyper-parameter
    parser.add_argument('--ifgpu', type=bool, default=True)
    parser.add_argument('--epoch_num', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1)

    args = parser.parse_args()

    t = BertTokenizer.from_pretrained(args.bert_model_type)

    main(args, t)
