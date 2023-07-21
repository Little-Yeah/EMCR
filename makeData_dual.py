# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from copy import copy


class dual_sample(object):
    def __init__(self,
                 original_sample,
                 text,
                 context,
                 forward_querys,
                 forward_answers,
                 backward_querys,
                 backward_answers,
                 sentiment_querys,
                 sentiment_answers,
                 cate_querys,
                 cate_answers,
                 category_querys,
                 category_answers,
                 attribute_querys,
                 attribute_answers):
        self.original_sample = original_sample
        self.text = text  #
        self.context = context
        self.forward_querys = forward_querys
        self.forward_answers = forward_answers
        self.backward_querys = backward_querys
        self.backward_answers = backward_answers
        self.sentiment_querys = sentiment_querys
        self.sentiment_answers = sentiment_answers
        self.cate_querys = cate_querys,
        self.cate_answers = cate_answers
        self.category_querys = category_querys
        self.category_answers = category_answers
        self.attribute_querys = attribute_querys
        self.attribute_answers = attribute_answers       


class sample_tokenized(object):
    def __init__(self,
                 original_sample,
                 contexts,
                 forward_querys,
                 forward_answers,
                 backward_querys,
                 backward_answers,
                 sentiment_querys,
                 sentiment_answers,
                 cate_querys,
                 cate_answers,
                 category_querys,
                 category_answers,
                 attribute_querys,
                 attribute_answers,
                 forward_seg,
                 backward_seg,
                 sentiment_seg,
                 cate_seg,
                 category_seg,
                 attribute_seg):
        self.original_sample = original_sample
        self.contexts = contexts
        self.forward_querys = forward_querys
        self.forward_answers = forward_answers
        self.backward_querys = backward_querys
        self.backward_answers = backward_answers
        self.sentiment_querys = sentiment_querys
        self.sentiment_answers = sentiment_answers
        self.category_querys = category_querys
        self.category_answers = category_answers
        self.attribute_querys = attribute_querys
        self.attribute_answers = attribute_answers       
        self.cate_querys = cate_querys
        self.cate_answers = cate_answers
        self.forward_seg = forward_seg
        self.backward_seg = backward_seg
        self.sentiment_seg = sentiment_seg
        self.cate_seg = cate_seg
        self.category_seg = category_seg
        self.attribute_seg = attribute_seg


class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self._context = pre_data['_context']
        self._forward_asp_query = pre_data['_forward_asp_query']
        self._forward_opi_query = pre_data['_forward_opi_query']  # [max_aspect_num, max_opinion_query_length]
        self._forward_asp_answer_start = pre_data['_forward_asp_answer_start']
        self._forward_asp_answer_end = pre_data['_forward_asp_answer_end']
        self._forward_opi_answer_start = pre_data['_forward_opi_answer_start']
        self._forward_opi_answer_end = pre_data['_forward_opi_answer_end']
        self._forward_asp_query_mask = pre_data['_forward_asp_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._forward_opi_query_mask = pre_data['_forward_opi_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._forward_asp_query_seg = pre_data['_forward_asp_query_seg']  # [max_aspect_num, max_opinion_query_length]
        self._forward_opi_query_seg = pre_data['_forward_opi_query_seg']  # [max_aspect_num, max_opinion_query_length]

        self._backward_asp_query = pre_data['_backward_asp_query']
        self._backward_opi_query = pre_data['_backward_opi_query']  # [max_aspect_num, max_opinion_query_length]
        self._backward_asp_answer_start = pre_data['_backward_asp_answer_start']
        self._backward_asp_answer_end = pre_data['_backward_asp_answer_end']
        self._backward_opi_answer_start = pre_data['_backward_opi_answer_start']
        self._backward_opi_answer_end = pre_data['_backward_opi_answer_end']
        self._backward_asp_query_mask = pre_data[
            '_backward_asp_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._backward_opi_query_mask = pre_data[
            '_backward_opi_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._backward_asp_query_seg = pre_data['_backward_asp_query_seg']  # [max_aspect_num, max_opinion_query_length]
        self._backward_opi_query_seg = pre_data['_backward_opi_query_seg']  # [max_aspect_num, max_opinion_query_length]

        self._sentiment_query = pre_data['_sentiment_query']  # [max_aspect_num, max_sentiment_query_length]
        self._sentiment_answer = pre_data['_sentiment_answer']
        self._sentiment_query_mask = pre_data['_sentiment_query_mask']  # [max_aspect_num, max_sentiment_query_length]
        self._sentiment_query_seg = pre_data['_sentiment_query_seg']  # [max_aspect_num, max_sentiment_query_length]

        self._cate_query = pre_data['_cate_query']  # [max_aspect_num, max_sentiment_query_length]
        self._cate_answer = pre_data['_cate_answer']
        self._cate_query_mask = pre_data['_cate_query_mask']  # [max_aspect_num, max_sentiment_query_length]
        self._cate_query_seg = pre_data['_cate_query_seg']  # [max_aspect_num, max_sentiment_query_length]

        self._category_query = pre_data['_category_query']  # [max_aspect_num, max_sentiment_query_length]
        self._category_answer = pre_data['_category_answer']
        self._category_query_mask = pre_data['_category_query_mask']  # [max_aspect_num, max_sentiment_query_length]
        self._category_query_seg = pre_data['_category_query_seg']  # [max_aspect_num, max_sentiment_query_length]

        self._attribute_query = pre_data['_attribute_query']  # [max_aspect_num, max_sentiment_query_length]
        self._attribute_answer = pre_data['_attribute_answer']
        self._attribute_query_mask = pre_data['_attribute_query_mask']  # [max_aspect_num, max_sentiment_query_length]
        self._attribute_query_seg = pre_data['_attribute_query_seg']  # [max_aspect_num, max_sentiment_query_length]

        self._aspect_num = pre_data['_aspect_num']
        self._opinion_num = pre_data['_opinion_num']




def pre_processing(sample_list, max_len):

    _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _forward_asp_query = []
    _forward_opi_query = []
    _forward_asp_answer_start = []
    _forward_asp_answer_end = []
    _forward_opi_answer_start = []
    _forward_opi_answer_end = []
    _forward_asp_query_mask = []
    _forward_opi_query_mask = []
    _forward_asp_query_seg = []
    _forward_opi_query_seg = []
    _backward_asp_query = []
    _backward_opi_query = []
    _backward_asp_answer_start = []
    _backward_asp_answer_end = []
    _backward_opi_answer_start = []
    _backward_opi_answer_end = []
    _backward_asp_query_mask = []
    _backward_opi_query_mask = []
    _backward_asp_query_seg = []
    _backward_opi_query_seg = []

    _sentiment_query = []
    _sentiment_answer = []
    _sentiment_query_mask = []
    _sentiment_query_seg = []

    _cate_query = []
    _cate_answer = []
    _cate_query_mask = []
    _cate_query_seg = []

    _category_query =[]
    _category_answer = []
    _category_query_mask =[]
    _category_query_seg = []

    _attribute_query =[]
    _attribute_answer = []
    _attribute_query_mask = []
    _attribute_query_seg = []

    _aspect_num = []
    _opinion_num = []

    _context = []

    for instance in sample_list:
        context_list = instance.contexts
        f_query_list = instance.forward_querys
        f_answer_list = instance.forward_answers
        f_query_seg_list = instance.forward_seg
        b_query_list = instance.backward_querys
        b_answer_list = instance.backward_answers
        b_query_seg_list = instance.backward_seg
        s_query_list = instance.sentiment_querys
        s_answer_list = instance.sentiment_answers
        s_query_seg_list = instance.sentiment_seg
        c_query_list = instance.cate_querys
        c_answer_list = instance.cate_answers
        c_query_seg_list = instance.cate_seg
        ca_query_list = instance.category_querys
        ca_answer_list = instance.category_answers
        ca_query_seg_list = instance.category_seg
        a_query_list = instance.attribute_querys
        a_answer_list = instance.attribute_answers
        a_query_seg_list = instance.attribute_seg        

        # _aspect_num: 1/2/3/...
        _aspect_num.append(int(len(f_query_list) - 1))
        _opinion_num.append(int(len(b_query_list) - 1))

        # context_pad_num = max_len['max_context_len'] -len(context_list[0])
        # _context.append(_tokenizer.convert_tokens_to_ids(
        #     [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in context_list[0]]))
        # _context[-1].extend([0] * context_pad_num)

        for j in range(len(context_list)):
            context_pad_num = max_len['max_context_len'] -len(context_list[0])
            _context.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in context_list[j]]))
            _context[-1].extend([0] * context_pad_num)


        # Forward
        # Aspect
        # query
        assert len(f_query_list[0]) == len(f_answer_list[0][0]) == len(f_answer_list[0][1])
        f_asp_pad_num = max_len['mfor_asp_len'] - len(f_query_list[0])

        _forward_asp_query.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[0]]))
        _forward_asp_query[-1].extend([0] * f_asp_pad_num)

        # query_mask
        _forward_asp_query_mask.append([1 for i in range(len(f_query_list[0]))])
        _forward_asp_query_mask[-1].extend([0] * f_asp_pad_num)

        # answer
        _forward_asp_answer_start.append(f_answer_list[0][0])
        _forward_asp_answer_start[-1].extend([-1] * f_asp_pad_num)
        _forward_asp_answer_end.append(f_answer_list[0][1])
        _forward_asp_answer_end[-1].extend([-1] * f_asp_pad_num)

        # seg
        _forward_asp_query_seg.append(f_query_seg_list[0])
        _forward_asp_query_seg[-1].extend([1] * f_asp_pad_num)

        # Opinion
        single_opinion_query = []
        single_opinion_query_mask = []
        single_opinion_query_seg = []
        single_opinion_answer_start = []
        single_opinion_answer_end = []
        for i in range(1, len(f_query_list)):
            assert len(f_query_list[i]) == len(f_answer_list[i][0]) == len(f_answer_list[i][1])
            pad_num = max_len['mfor_opi_len'] - len(f_query_list[i])

            
            # query
            single_opinion_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[i]]))
            single_opinion_query[-1].extend([0] * pad_num)

            # query_mask
            single_opinion_query_mask.append([1 for i in range(len(f_query_list[i]))])
            single_opinion_query_mask[-1].extend([0] * pad_num)

            # query_seg
            single_opinion_query_seg.append(f_query_seg_list[i])
            single_opinion_query_seg[-1].extend([1] * pad_num)

            # answer
            single_opinion_answer_start.append(f_answer_list[i][0])
            single_opinion_answer_start[-1].extend([-1] * pad_num)
            single_opinion_answer_end.append(f_answer_list[i][1])
            single_opinion_answer_end[-1].extend([-1] * pad_num)

        # PAD: max_aspect_num
        _forward_opi_query.append(single_opinion_query)
        _forward_opi_query[-1].extend([[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_query_mask.append(single_opinion_query_mask)
        _forward_opi_query_mask[-1].extend([[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_query_seg.append(single_opinion_query_seg)
        _forward_opi_query_seg[-1].extend([[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_answer_start.append(single_opinion_answer_start)
        _forward_opi_answer_start[-1].extend([[-1 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))
        _forward_opi_answer_end.append(single_opinion_answer_end)
        _forward_opi_answer_end[-1].extend([[-1 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        # Backward
        # opinion
        # query
        assert len(b_query_list[0]) == len(b_answer_list[0][0]) == len(b_answer_list[0][1])
        b_opi_pad_num = max_len['mback_opi_len'] - len(b_query_list[0])

        _backward_opi_query.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[0]]))
        _backward_opi_query[-1].extend([0] * b_opi_pad_num)

        # mask
        _backward_opi_query_mask.append([1 for i in range(len(b_query_list[0]))])
        _backward_opi_query_mask[-1].extend([0] * b_opi_pad_num)

        # answer
        _backward_opi_answer_start.append(b_answer_list[0][0])
        _backward_opi_answer_start[-1].extend([-1] * b_opi_pad_num)
        _backward_opi_answer_end.append(b_answer_list[0][1])
        _backward_opi_answer_end[-1].extend([-1] * b_opi_pad_num)

        # seg
        _backward_opi_query_seg.append(b_query_seg_list[0])
        _backward_opi_query_seg[-1].extend([1] * b_opi_pad_num)

        # Aspect
        single_aspect_query = []
        single_aspect_query_mask = []
        single_aspect_query_seg = []
        single_aspect_answer_start = []
        single_aspect_answer_end = []
        for i in range(1, len(b_query_list)):
            assert len(b_query_list[i]) == len(b_answer_list[i][0]) == len(b_answer_list[i][1])
            pad_num = max_len['mback_asp_len'] - len(b_query_list[i])
            # query
            single_aspect_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[i]]))
            single_aspect_query[-1].extend([0] * pad_num)

            # query_mask
            single_aspect_query_mask.append([1 for i in range(len(b_query_list[i]))])
            single_aspect_query_mask[-1].extend([0] * pad_num)

            # query_seg
            single_aspect_query_seg.append(b_query_seg_list[i])
            single_aspect_query_seg[-1].extend([1] * pad_num)

            # answer
            single_aspect_answer_start.append(b_answer_list[i][0])
            single_aspect_answer_start[-1].extend([-1] * pad_num)
            single_aspect_answer_end.append(b_answer_list[i][1])
            single_aspect_answer_end[-1].extend([-1] * pad_num)

        # PAD: max_opinion_num
        _backward_asp_query.append(single_aspect_query)
        _backward_asp_query[-1].extend([[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_query_mask.append(single_aspect_query_mask)
        _backward_asp_query_mask[-1].extend([[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_query_seg.append(single_aspect_query_seg)
        _backward_asp_query_seg[-1].extend([[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_answer_start.append(single_aspect_answer_start)
        _backward_asp_answer_start[-1].extend([[-1 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))
        _backward_asp_answer_end.append(single_aspect_answer_end)
        _backward_asp_answer_end[-1].extend([[-1 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        # Sentiment
        single_sentiment_query = []
        single_sentiment_query_mask = []
        single_sentiment_query_seg = []
        single_sentiment_answer = []
        for j in range(len(s_query_list)):
            sent_pad_num = max_len['max_sent_len'] - len(s_query_list[j])
            single_sentiment_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in s_query_list[j]]))
            single_sentiment_query[-1].extend([0] * sent_pad_num)

            single_sentiment_query_mask.append([1 for i in range(len(s_query_list[j]))])
            single_sentiment_query_mask[-1].extend([0] * sent_pad_num)

            # query_seg
            single_sentiment_query_seg.append(s_query_seg_list[j])
            single_sentiment_query_seg[-1].extend([1] * sent_pad_num)

            single_sentiment_answer.append(s_answer_list[j])

        _sentiment_query.append(single_sentiment_query)
        _sentiment_query[-1].extend([[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_mask.append(single_sentiment_query_mask)
        _sentiment_query_mask[-1].extend([[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_seg.append(single_sentiment_query_seg)
        _sentiment_query_seg[-1].extend([[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_answer.append(single_sentiment_answer)
        _sentiment_answer[-1].extend([-1] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        # Category-OVERALL
        single_cate_query = []
        single_cate_query_mask = []
        single_cate_query_seg = []
        single_cate_answer = []
        for j in range(len(c_query_list)):
            cate_pad_num = max_len['max_cate_len'] - len(c_query_list[j])
            single_cate_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in c_query_list[j]]))
            single_cate_query[-1].extend([0] * cate_pad_num)

            single_cate_query_mask.append([1 for i in range(len(c_query_list[j]))])
            single_cate_query_mask[-1].extend([0] * cate_pad_num)

            # query_seg
            single_cate_query_seg.append(c_query_seg_list[j])
            single_cate_query_seg[-1].extend([1] * cate_pad_num)

            single_cate_answer.append(c_answer_list[j])

        _cate_query.append(single_cate_query)
        _cate_query[-1].extend([[0 for i in range(max_len['max_cate_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _cate_query_mask.append(single_cate_query_mask)
        _cate_query_mask[-1].extend([[0 for i in range(max_len['max_cate_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _cate_query_seg.append(single_cate_query_seg)
        _cate_query_seg[-1].extend([[0 for i in range(max_len['max_cate_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _cate_answer.append(single_cate_answer)
        _cate_answer[-1].extend([-1] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        # Category-1
        single_category_query = []
        single_category_query_mask = []
        single_category_query_seg = []
        single_category_answer = []
        for j in range(len(ca_query_list)):
            category_pad_num = max_len['max_category_len'] - len(ca_query_list[j])
            single_category_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in ca_query_list[j]]))
            single_category_query[-1].extend([0] * category_pad_num)

            single_category_query_mask.append([1 for i in range(len(ca_query_list[j]))])
            single_category_query_mask[-1].extend([0] * category_pad_num)

            # query_seg
            single_category_query_seg.append(ca_query_seg_list[j])
            single_category_query_seg[-1].extend([1] * category_pad_num)

            single_category_answer.append(ca_answer_list[j])

        _category_query.append(single_category_query)
        _category_query[-1].extend([[0 for i in range(max_len['max_category_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _category_query_mask.append(single_category_query_mask)
        _category_query_mask[-1].extend([[0 for i in range(max_len['max_category_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _category_query_seg.append(single_category_query_seg)
        _category_query_seg[-1].extend([[0 for i in range(max_len['max_category_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _category_answer.append(single_category_answer)
        _category_answer[-1].extend([-1] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        # Category-
        single_attribute_query = []
        single_attribute_query_mask = []
        single_attribute_query_seg = []
        single_attribute_answer = []
        for j in range(len(a_query_list)):
            attribute_pad_num = max_len['max_attribute_len'] - len(a_query_list[j])
            single_attribute_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in a_query_list[j]]))
            single_attribute_query[-1].extend([0] * attribute_pad_num)

            single_attribute_query_mask.append([1 for i in range(len(a_query_list[j]))])
            single_attribute_query_mask[-1].extend([0] * attribute_pad_num)

            # query_seg
            single_attribute_query_seg.append(a_query_seg_list[j])
            single_attribute_query_seg[-1].extend([1] * attribute_pad_num)

            single_attribute_answer.append(a_answer_list[j])

        _attribute_query.append(single_attribute_query)
        _attribute_query[-1].extend([[0 for i in range(max_len['max_attribute_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _attribute_query_mask.append(single_attribute_query_mask)
        _attribute_query_mask[-1].extend([[0 for i in range(max_len['max_attribute_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _attribute_query_seg.append(single_attribute_query_seg)
        _attribute_query_seg[-1].extend([[0 for i in range(max_len['max_attribute_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _attribute_answer.append(single_attribute_answer)
        _attribute_answer[-1].extend([-1] * (max_len['max_aspect_num'] - _aspect_num[-1]))

    result = {"_forward_asp_query":_forward_asp_query, "_forward_opi_query":_forward_opi_query,
              "_forward_asp_answer_start":_forward_asp_answer_start, "_forward_asp_answer_end":_forward_asp_answer_end,
              "_forward_opi_answer_start":_forward_opi_answer_start, "_forward_opi_answer_end":_forward_opi_answer_end,
              "_forward_asp_query_mask":_forward_asp_query_mask, "_forward_opi_query_mask":_forward_opi_query_mask,
              "_forward_asp_query_seg":_forward_asp_query_seg, "_forward_opi_query_seg":_forward_opi_query_seg,
              "_backward_asp_query":_backward_asp_query, "_backward_opi_query":_backward_opi_query,
              "_backward_asp_answer_start":_backward_asp_answer_start, "_backward_asp_answer_end":_backward_asp_answer_end,
              "_backward_opi_answer_start":_backward_opi_answer_start, "_backward_opi_answer_end":_backward_opi_answer_end,
              "_backward_asp_query_mask":_backward_asp_query_mask, "_backward_opi_query_mask":_backward_opi_query_mask,
              "_backward_asp_query_seg":_backward_asp_query_seg, "_backward_opi_query_seg":_backward_opi_query_seg,
              "_sentiment_query":_sentiment_query, "_sentiment_answer":_sentiment_answer, 
              "_sentiment_query_mask":_sentiment_query_mask, "_sentiment_query_seg":_sentiment_query_seg,
              "_cate_query":_cate_query, "_cate_answer":_cate_answer, 
              "_cate_query_mask":_cate_query_mask, "_cate_query_seg":_cate_query_seg,
              "_category_query":_category_query,"_category_answer":_category_answer,
              "_category_query_mask":_category_query_mask, "_category_query_seg":_category_query_seg,
              "_attribute_query":_attribute_query,"_attribute_answer":_attribute_answer,
              "_attribute_query_mask":_attribute_query_mask, "_attribute_query_seg":_attribute_query_seg,
              "_aspect_num":_aspect_num, "_opinion_num":_opinion_num, "_context":_context}
    return OriginalDataset(result)


def tokenized_data(data):
    max_context_length = 0
    max_forward_asp_query_length = 0
    max_forward_opi_query_length = 0
    max_backward_asp_query_length = 0
    max_backward_opi_query_length = 0
    max_sentiment_query_length = 0
    max_cate_query_length = 0
    max_category_query_length = 0
    max_attribute_query_length = 0
    max_aspect_num = 0
    max_opinion_num = 0
    tokenized_sample_list = []
    for sample in data:
        contexts = []
        forward_querys = []
        forward_querys_only = []
        forward_answers = []
        backward_querys = []
        backward_querys_only = []
        backward_answers = []
        sentiment_querys = []
        sentiment_answers = []
        cate_querys = []
        cate_answers = []
        category_querys = []
        category_answers = []
        attribute_querys = []
        attribute_answers = []
        forward_querys_seg = []
        backward_querys_seg = []
        sentiment_querys_seg = []
        cate_querys_seg = []
        category_querys_seg = []
        attribute_querys_seg = []
        for idx in range(len(sample.context)):
            temp_context = sample.context
            if len(temp_context) > max_context_length:
                max_context_length = len(temp_context)
        if int(len(sample.forward_querys) - 1) > max_aspect_num:
            max_aspect_num = int(len(sample.forward_querys) - 1)
        if int(len(sample.backward_querys) - 1) > max_opinion_num:
            max_opinion_num = int(len(sample.backward_querys) - 1)
        for idx in range(len(sample.forward_querys)):
            temp_query = sample.forward_querys[idx]
            temp_text = sample.text
            temp_context = sample.context
            temp_answer = sample.forward_answers[idx]
            # temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + ['[PAD]']*len(temp_text)
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * (len(temp_text))
            temp_answer[0] = [-1] * (len(temp_query) + 2) + temp_answer[0]
            temp_answer[1] = [-1] * (len(temp_query) + 2) + temp_answer[1]
            assert len(temp_answer[0]) == len(temp_answer[1]) == len(temp_query_to) == len(temp_query_seg)
            if idx == 0:
                if len(temp_query_to) > max_forward_asp_query_length:
                    max_forward_asp_query_length = len(temp_query_to)
            else:
                if len(temp_query_to) > max_forward_opi_query_length:
                    max_forward_opi_query_length = len(temp_query_to)
            contexts.append(temp_context)
            forward_querys.append(temp_query_to)
            forward_answers.append(temp_answer)
            forward_querys_seg.append(temp_query_seg)
        for idx in range(len(sample.backward_querys)):
            temp_query = sample.backward_querys[idx]
            temp_text = sample.text
            temp_answer = sample.backward_answers[idx]
            # temp_query_only = ['[CLS]'] + temp_query + ['[SEP]'] + ['[PAD]']*len(temp_text)
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * (len(temp_text))
            temp_answer[0] = [-1] * (len(temp_query) + 2) + temp_answer[0]
            temp_answer[1] = [-1] * (len(temp_query) + 2) + temp_answer[1]
            assert len(temp_answer[0]) == len(temp_answer[1]) == len(temp_query_to) == len(temp_query_seg)
            if idx == 0:
                if len(temp_query_to) > max_backward_opi_query_length:
                    max_backward_opi_query_length = len(temp_query_to)
            else:
                if len(temp_query_to) > max_backward_asp_query_length:
                    max_backward_asp_query_length = len(temp_query_to)
            backward_querys.append(temp_query_to)
            # backward_querys_only.append(temp_query_only)
            backward_answers.append(temp_answer)
            backward_querys_seg.append(temp_query_seg)
        for idx in range(len(sample.sentiment_querys)):
            temp_query = sample.sentiment_querys[idx]
            temp_text = sample.text
            temp_answer = sample.sentiment_answers[idx]
            # if if_tokenized:
            #     print(2)
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            assert len(temp_query_to) == len(temp_query_seg)
            if len(temp_query_to) > max_sentiment_query_length:
                max_sentiment_query_length = len(temp_query_to)
            sentiment_querys.append(temp_query_to)
            sentiment_answers.append(temp_answer)
            sentiment_querys_seg.append(temp_query_seg)

        for idx in range(len(sample.cate_querys)):
            temp_query = sample.cate_querys[idx]
            temp_text = sample.text
            temp_answer = sample.cate_answers[idx]
            # if if_tokenized:
            #     print(2)
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            assert len(temp_query_to) == len(temp_query_seg)
            if len(temp_query_to) > max_cate_query_length:
                max_cate_query_length = len(temp_query_to)
            cate_querys.append(temp_query_to)
            cate_answers.append(temp_answer)
            cate_querys_seg.append(temp_query_seg)

        for idx in range(len(sample.category_querys)):
            temp_query = sample.category_querys[idx]
            temp_text = sample.text
            temp_answer = sample.category_answers[idx]
            # if if_tokenized:
            #     print(2)
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            assert len(temp_query_to) == len(temp_query_seg)
            if len(temp_query_to) > max_category_query_length:
                max_category_query_length = len(temp_query_to)
            category_querys.append(temp_query_to)
            category_answers.append(temp_answer)
            category_querys_seg.append(temp_query_seg)  

        for idx in range(len(sample.attribute_querys)):
            temp_query = sample.attribute_querys[idx]
            temp_text = sample.text
            temp_answer = sample.attribute_answers[idx]
            # if if_tokenized:
            #     print(2)
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            assert len(temp_query_to) == len(temp_query_seg)
            if len(temp_query_to) > max_attribute_query_length:
                max_attribute_query_length = len(temp_query_to)
            attribute_querys.append(temp_query_to)
            attribute_answers.append(temp_answer)
            attribute_querys_seg.append(temp_query_seg)
        # print('tokenized_data-contexts',contexts)                      
        temp_sample = sample_tokenized(sample.original_sample, contexts, forward_querys,forward_answers, 
                                       backward_querys, backward_answers, sentiment_querys, sentiment_answers,cate_querys, cate_answers,
                                       category_querys, category_answers, attribute_querys, attribute_answers,
                                       forward_querys_seg, backward_querys_seg, sentiment_querys_seg, cate_querys_seg,
                                       category_querys_seg, attribute_querys_seg)
        tokenized_sample_list.append(temp_sample)

        max_query_length = max(max_forward_asp_query_length,max_forward_opi_query_length,max_backward_asp_query_length,max_backward_opi_query_length)

    return tokenized_sample_list, {'mfor_asp_len': max_query_length,
                                   'mfor_opi_len': max_query_length,
                                   'mback_asp_len': max_query_length,
                                   'mback_opi_len': max_query_length,
                                   'max_sent_len': max_sentiment_query_length,
                                   'max_cate_len':max_cate_query_length,
                                   'max_category_len':max_category_query_length,
                                   'max_attribute_len':max_attribute_query_length,
                                   'max_aspect_num': max_aspect_num,
                                   'max_opinion_num': max_opinion_num,
                                   'max_context_len':max_context_length}


if __name__ == '__main__':
    for dataset_name in ['Laptop_ACOS']:
        output_path = './data/preprocess/' + dataset_name + '.pt'
        train_data = torch.load("./data/preprocess/" + dataset_name + "_train_dual.pt")
        dev_data = torch.load("./data/preprocess/" + dataset_name + "_dev_dual.pt")
        test_data = torch.load("./data/preprocess/" + dataset_name + "_test_dual.pt")
        
        train_tokenized, train_max_len = tokenized_data(train_data)
        dev_tokenized, dev_max_len = tokenized_data(dev_data)
        test_tokenized, test_max_len = tokenized_data(test_data)
        max_len = train_max_len.copy()
        for key in max_len:
            max_len[key] = max(train_max_len[key],dev_max_len[key],test_max_len[key])

        print('preprocessing_data')

        train_preprocess = pre_processing(train_tokenized,max_len)
        dev_preprocess = pre_processing(dev_tokenized, max_len)
        test_preprocess = pre_processing(test_tokenized, max_len)
        print('save_data')
        torch.save({'train': train_preprocess, 'dev': dev_preprocess, 'test': test_preprocess}, output_path)
