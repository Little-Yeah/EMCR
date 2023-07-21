import pickle
import sys
import torch
# #####REST15###########
# aspect_cate_list = ['location general',
#                     'food prices',
#                     'food quality',
#                     'food general',
#                     'ambience general',
#                     'service general',
#                     'restaurant prices',
#                     'drinks prices',
#                     'restaurant miscellaneous',
#                     'drinks quality',
#                     'drinks style_options',
#                     'restaurant general',
#                     'food style_options']

# #########laptop###########
# aspect_cate_list = ['MULTIMEDIA_DEVICES#PRICE', 'OS#QUALITY', 'SHIPPING#QUALITY', 'GRAPHICS#OPERATION_PERFORMANCE', 'CPU#OPERATION_PERFORMANCE', 
#             'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'SHIPPING#PRICE', 'POWER_SUPPLY#CONNECTIVITY', 'SOFTWARE#USABILITY', 
#             'FANS&COOLING#GENERAL', 'GRAPHICS#DESIGN_FEATURES', 'BATTERY#GENERAL', 'HARD_DISC#USABILITY', 'FANS&COOLING#DESIGN_FEATURES', 
#             'MEMORY#DESIGN_FEATURES', 'MOUSE#USABILITY', 'CPU#GENERAL', 'LAPTOP#QUALITY', 'POWER_SUPPLY#GENERAL', 'PORTS#QUALITY', 
#             'KEYBOARD#PORTABILITY', 'SUPPORT#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#USABILITY', 'MOUSE#GENERAL', 'KEYBOARD#MISCELLANEOUS', 
#             'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OS#MISCELLANEOUS', 'LAPTOP#MISCELLANEOUS', 'SOFTWARE#PRICE', 'FANS&COOLING#OPERATION_PERFORMANCE', 
#             'MEMORY#QUALITY', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'HARD_DISC#GENERAL', 'MEMORY#GENERAL', 'DISPLAY#OPERATION_PERFORMANCE', 
#             'MULTIMEDIA_DEVICES#GENERAL', 'LAPTOP#GENERAL', 'MOTHERBOARD#QUALITY', 'LAPTOP#PORTABILITY', 'KEYBOARD#PRICE', 'SUPPORT#OPERATION_PERFORMANCE', 
#             'GRAPHICS#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'DISPLAY#GENERAL', 'BATTERY#QUALITY', 'LAPTOP#USABILITY', 'LAPTOP#DESIGN_FEATURES', 
#             'PORTS#CONNECTIVITY', 'HARDWARE#QUALITY', 'SUPPORT#GENERAL', 'MOTHERBOARD#GENERAL', 'PORTS#USABILITY', 'KEYBOARD#QUALITY', 'GRAPHICS#USABILITY', 
#             'HARD_DISC#PRICE', 'OPTICAL_DRIVES#USABILITY', 'MULTIMEDIA_DEVICES#CONNECTIVITY', 'HARDWARE#DESIGN_FEATURES', 'MEMORY#USABILITY', 
#             'SHIPPING#GENERAL', 'CPU#PRICE', 'Out_Of_Scope#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#QUALITY', 'OS#PRICE', 'SUPPORT#QUALITY', 
#             'OPTICAL_DRIVES#GENERAL', 'HARDWARE#USABILITY', 'DISPLAY#DESIGN_FEATURES', 'PORTS#GENERAL', 'COMPANY#OPERATION_PERFORMANCE', 
#             'COMPANY#GENERAL', 'Out_Of_Scope#GENERAL', 'KEYBOARD#DESIGN_FEATURES', 'Out_Of_Scope#OPERATION_PERFORMANCE', 
#             'OPTICAL_DRIVES#DESIGN_FEATURES', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#USABILITY', 'DISPLAY#USABILITY', 'POWER_SUPPLY#QUALITY', 
#             'HARD_DISC#DESIGN_FEATURES', 'DISPLAY#QUALITY', 'MOUSE#DESIGN_FEATURES', 'COMPANY#QUALITY', 'HARDWARE#GENERAL', 'COMPANY#PRICE', 
#             'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE', 'SOFTWARE#PORTABILITY', 'HARD_DISC#OPERATION_PERFORMANCE', 
#             'BATTERY#DESIGN_FEATURES', 'CPU#QUALITY', 'WARRANTY#GENERAL', 'OS#DESIGN_FEATURES', 'OS#OPERATION_PERFORMANCE', 'OS#USABILITY', 
#             'SOFTWARE#GENERAL', 'SUPPORT#PRICE', 'SHIPPING#OPERATION_PERFORMANCE', 'DISPLAY#PRICE', 'LAPTOP#PRICE', 'OS#GENERAL', 'HARDWARE#PRICE', 
#             'SOFTWARE#DESIGN_FEATURES', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY', 'FANS&COOLING#QUALITY', 'BATTERY#OPERATION_PERFORMANCE', 
#             'CPU#DESIGN_FEATURES', 'PORTS#OPERATION_PERFORMANCE', 'SOFTWARE#OPERATION_PERFORMANCE', 'KEYBOARD#GENERAL', 'SOFTWARE#QUALITY', 
#             'LAPTOP#CONNECTIVITY', 'POWER_SUPPLY#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE', 'WARRANTY#QUALITY', 'HARD_DISC#QUALITY', 
#             'POWER_SUPPLY#OPERATION_PERFORMANCE', 'PORTS#DESIGN_FEATURES', 'Out_Of_Scope#USABILITY']
# category_list = ['BATTERY', 'COMPANY', 'CPU', 'DISPLAY', 'FANS&COOLING', 'GRAPHICS', 'HARDWARE', 'HARD_DISC', 'KEYBOARD', 'LAPTOP', 'MEMORY', 'MOTHERBOARD', 
#                 'MOUSE', 'MULTIMEDIA_DEVICES', 'OPTICAL_DRIVES', 'OS', 'Out_Of_Scope', 'PORTS', 'POWER_SUPPLY', 'SHIPPING', 'SOFTWARE', 'SUPPORT', 'WARRANTY']
# attribute_list = ['DESIGN_FEATURES', 'GENERAL', 'OPERATION_PERFORMANCE', 'QUALITY', 'PRICE', 'USABILITY', 'MISCELLANEOUS', 'PORTABILITY', 'CONNECTIVITY']

aspect_cate_list = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES', 
            'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
category_list = ['RESTAURANT','SERVICE','FOOD','DRINKS','AMBIENCE','LOCATION']
attribute_list =['GENERAL','QUALITY','STYLE_OPTIONS','PRICES','MISCELLANEOUS']

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
        self.original_sample = original_sample  #
        self.text = text  #
        self.context =context
        self.forward_querys = forward_querys
        self.forward_answers = forward_answers
        self.backward_querys = backward_querys
        self.backward_answers = backward_answers
        self.sentiment_querys = sentiment_querys
        self.sentiment_answers = sentiment_answers
        self.cate_querys = cate_querys
        self.cate_answers = cate_answers
        self.category_querys = category_querys
        self.category_answers = category_answers
        self.attribute_querys = attribute_querys
        self.attribute_answers = attribute_answers


def get_text(line):
    temp = line.split('####')
    assert len(temp) == 2
    context = temp[0]
    word_list = temp.pop(0).split()
    return word_list,context


def get_quad(line):

    aspect_list = []
    opinion_list = []
    quad_aspect = []
    quad_opinion = []
    quad_sentiment = []
    quad_cate = []
    quad_category = []
    quad_attribute = []
    quadruple_list = []

    temp = line.split('####')
    word_list = temp[0].split()
    origin_sentence = temp.pop(0)
    quad_list = eval(temp[0])
    valid = True

    for quad in quad_list:
        if quad[0] == 'NULL':
            asp_start_idx = -1
            asp_end_idx = -1
            valid = False
        else:
            aspect = quad[0].split()
            if set(aspect) < set(word_list):
                asp_start_idx = word_list.index(aspect[0])
                asp_end_idx = word_list.index(aspect[-1])
            else:
                valid = False
                continue
        asp_span = [asp_start_idx, asp_end_idx]
        aspect_list.append(asp_span)

        if quad[3] == 'NULL':
            opi_start_idx = -1
            opi_end_idx = -1
            valid = False
        else:
            opinion = quad[3].split()
            if set(opinion) < set(word_list):
                opi_start_idx = word_list.index(opinion[0])
                opi_end_idx = word_list.index(opinion[-1])
            else:
                valid = False
                continue
        opi_span = [opi_start_idx, opi_end_idx]
        opinion_list.append(opi_span)

        if quad[2] == 'positive':
            polarity = 1
        elif quad[2] == 'negative':
            polarity = 2
        else:
            polarity = 0


        cate = aspect_cate_list.index(quad[1])
        quad_cate.append(cate)

        category = quad[1].split('#')[0]
        category = category_list.index(category)
        
        attribute = quad[1].split('#')[1]
        attribute = attribute_list.index(attribute) 

        quadruple = [asp_span, opi_span, polarity,cate, category, attribute]
        quadruple_list.append(quadruple)

    return valid,  quadruple_list


def fusion_dual_quad(quadruple):
    quad_aspect = []
    quad_opinion = []
    quad_sentiment = []
    quad_cate = []
    quad_category = []
    quad_attribute = []
    dual_opinion = []
    dual_aspect = []
    valid = True

    for q in quadruple:
        if q[0] not in quad_aspect:
            quad_aspect.append(q[0])
            quad_opinion.append([q[1]])
            quad_sentiment.append(q[2])
            quad_cate.append(q[3])
            quad_category.append(q[4])
            quad_attribute.append(q[5])
        else:
            idx = quad_aspect.index(q[0])
            quad_opinion[idx].append(q[1])
            if quad_sentiment[idx] != q[2] or quad_cate[idx] !=q[3]:
                valid = False
                break
        if q[1] not in dual_opinion:
            dual_opinion.append(q[1])
            dual_aspect.append([q[0]])
        else:
            idx = dual_opinion.index(q[1])
            dual_aspect[idx].append(q[0])

    return valid, quad_aspect, quad_opinion, quad_sentiment, quad_cate, dual_opinion, dual_aspect, quad_category,quad_attribute


if __name__ == '__main__':
    home_path = './data/original/'
    # dataset_name_list = ['rest15', 'rest16','Laptop_ACOS','Restaurant_ACOS']
    dataset_name_list = ['Restaurant_ACOS']
    dataset_type_list = ['train', 'test', 'dev']
    for dataset_name in dataset_name_list:
        # if dataset_name == 'Laptop_ACOS':
        #     aspect_cate_list = ['MULTIMEDIA_DEVICES#PRICE', 'OS#QUALITY', 'SHIPPING#QUALITY', 'GRAPHICS#OPERATION_PERFORMANCE', 'CPU#OPERATION_PERFORMANCE', 
        #     'COMPANY#DESIGN_FEATURES', 'MEMORY#OPERATION_PERFORMANCE', 'SHIPPING#PRICE', 'POWER_SUPPLY#CONNECTIVITY', 'SOFTWARE#USABILITY', 
        #     'FANS&COOLING#GENERAL', 'GRAPHICS#DESIGN_FEATURES', 'BATTERY#GENERAL', 'HARD_DISC#USABILITY', 'FANS&COOLING#DESIGN_FEATURES', 
        #     'MEMORY#DESIGN_FEATURES', 'MOUSE#USABILITY', 'CPU#GENERAL', 'LAPTOP#QUALITY', 'POWER_SUPPLY#GENERAL', 'PORTS#QUALITY', 
        #     'KEYBOARD#PORTABILITY', 'SUPPORT#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#USABILITY', 'MOUSE#GENERAL', 'KEYBOARD#MISCELLANEOUS', 
        #     'MULTIMEDIA_DEVICES#DESIGN_FEATURES', 'OS#MISCELLANEOUS', 'LAPTOP#MISCELLANEOUS', 'SOFTWARE#PRICE', 'FANS&COOLING#OPERATION_PERFORMANCE', 
        #     'MEMORY#QUALITY', 'OPTICAL_DRIVES#OPERATION_PERFORMANCE', 'HARD_DISC#GENERAL', 'MEMORY#GENERAL', 'DISPLAY#OPERATION_PERFORMANCE', 
        #     'MULTIMEDIA_DEVICES#GENERAL', 'LAPTOP#GENERAL', 'MOTHERBOARD#QUALITY', 'LAPTOP#PORTABILITY', 'KEYBOARD#PRICE', 'SUPPORT#OPERATION_PERFORMANCE', 
        #     'GRAPHICS#GENERAL', 'MOTHERBOARD#OPERATION_PERFORMANCE', 'DISPLAY#GENERAL', 'BATTERY#QUALITY', 'LAPTOP#USABILITY', 'LAPTOP#DESIGN_FEATURES', 
        #     'PORTS#CONNECTIVITY', 'HARDWARE#QUALITY', 'SUPPORT#GENERAL', 'MOTHERBOARD#GENERAL', 'PORTS#USABILITY', 'KEYBOARD#QUALITY', 'GRAPHICS#USABILITY', 
        #     'HARD_DISC#PRICE', 'OPTICAL_DRIVES#USABILITY', 'MULTIMEDIA_DEVICES#CONNECTIVITY', 'HARDWARE#DESIGN_FEATURES', 'MEMORY#USABILITY', 
        #     'SHIPPING#GENERAL', 'CPU#PRICE', 'Out_Of_Scope#DESIGN_FEATURES', 'MULTIMEDIA_DEVICES#QUALITY', 'OS#PRICE', 'SUPPORT#QUALITY', 
        #     'OPTICAL_DRIVES#GENERAL', 'HARDWARE#USABILITY', 'DISPLAY#DESIGN_FEATURES', 'PORTS#GENERAL', 'COMPANY#OPERATION_PERFORMANCE', 
        #     'COMPANY#GENERAL', 'Out_Of_Scope#GENERAL', 'KEYBOARD#DESIGN_FEATURES', 'Out_Of_Scope#OPERATION_PERFORMANCE', 
        #     'OPTICAL_DRIVES#DESIGN_FEATURES', 'LAPTOP#OPERATION_PERFORMANCE', 'KEYBOARD#USABILITY', 'DISPLAY#USABILITY', 'POWER_SUPPLY#QUALITY', 
        #     'HARD_DISC#DESIGN_FEATURES', 'DISPLAY#QUALITY', 'MOUSE#DESIGN_FEATURES', 'COMPANY#QUALITY', 'HARDWARE#GENERAL', 'COMPANY#PRICE', 
        #     'MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE', 'KEYBOARD#OPERATION_PERFORMANCE', 'SOFTWARE#PORTABILITY', 'HARD_DISC#OPERATION_PERFORMANCE', 
        #     'BATTERY#DESIGN_FEATURES', 'CPU#QUALITY', 'WARRANTY#GENERAL', 'OS#DESIGN_FEATURES', 'OS#OPERATION_PERFORMANCE', 'OS#USABILITY', 
        #     'SOFTWARE#GENERAL', 'SUPPORT#PRICE', 'SHIPPING#OPERATION_PERFORMANCE', 'DISPLAY#PRICE', 'LAPTOP#PRICE', 'OS#GENERAL', 'HARDWARE#PRICE', 
        #     'SOFTWARE#DESIGN_FEATURES', 'HARD_DISC#MISCELLANEOUS', 'PORTS#PORTABILITY', 'FANS&COOLING#QUALITY', 'BATTERY#OPERATION_PERFORMANCE', 
        #     'CPU#DESIGN_FEATURES', 'PORTS#OPERATION_PERFORMANCE', 'SOFTWARE#OPERATION_PERFORMANCE', 'KEYBOARD#GENERAL', 'SOFTWARE#QUALITY', 
        #     'LAPTOP#CONNECTIVITY', 'POWER_SUPPLY#DESIGN_FEATURES', 'HARDWARE#OPERATION_PERFORMANCE', 'WARRANTY#QUALITY', 'HARD_DISC#QUALITY', 
        #     'POWER_SUPPLY#OPERATION_PERFORMANCE', 'PORTS#DESIGN_FEATURES', 'Out_Of_Scope#USABILITY']
        # elif dataset_name == 'Restaurant_ACOS':
        #     aspect_cate_list = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#GENERAL', 'FOOD#QUALITY', 'FOOD#STYLE_OPTIONS', 'DRINKS#STYLE_OPTIONS', 'DRINKS#PRICES', 
        #     'AMBIENCE#GENERAL', 'RESTAURANT#PRICES', 'FOOD#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY', 'LOCATION#GENERAL']
        # else:
        #     aspect_cate_list = ['location general',
        #             'food prices',
        #             'food quality',
        #             'food general',
        #             'ambience general',
        #             'service general',
        #             'restaurant prices',
        #             'drinks prices',
        #             'restaurant miscellaneous',
        #             'drinks quality',
        #             'drinks style_options',
        #             'restaurant general',
        #             'food style_options']
        for dataset_type in dataset_type_list:
            quad_list_save = []
            output_path = "./data/preprocess/" + dataset_name + "_" + dataset_type + "_dual.pt"
            # read text
            f = open(home_path + dataset_name + "/" + dataset_type + ".txt", "r", encoding="utf-8")
            lines = f.readlines()
            f.close()
            sample_list = []
            for k in range(len(lines)):
                line = lines[k]
                text,context = get_text(line)
                valid, quad_list = get_quad(line)
                if valid == False:
                    continue
                valid, quad_aspect, quad_opinion, quad_sentiment, quad_cate, dual_opinion, dual_aspect,quad_category,quad_attribute = fusion_dual_quad(quad_list)
                if valid == False:
                    continue
                sub_quad_list = []
                for item in quad_list:
                    sub_quad_list.append(tuple(item))
                quad_list_save.append(sub_quad_list)
                forward_query_list = []
                backward_query_list = []
                sentiment_query_list = []
                cate_query_list = []
                forward_answer_list = []
                backward_answer_list = []
                sentiment_answer_list = []
                cate_answer_list = []
                category_query_list =[]
                category_answer_list = []
                attribute_query_list = []
                attribute_answer_list = []

                forward_query_list.append(["What", "aspects", "?"])
                start = [0] * len(text)
                end = [0] * len(text)
                for ta in quad_aspect:
                    start[ta[0]] = 1
                    end[ta[-1]] = 1
                forward_answer_list.append([start, end])

                backward_query_list.append(["What", "opinions", "?"])
                start = [0] * len(text)
                end = [0] * len(text)
                for to in dual_opinion:
                    start[to[0]] = 1
                    end[to[-1]] = 1
                backward_answer_list.append([start, end])

                for idx in range(len(quad_aspect)):
                    ta = quad_aspect[idx]
                    # opinion query
                    query = ["What", "opinion", "given", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["?"]
                    forward_query_list.append(query)

                    start = [0] * len(text)
                    end = [0] * len(text)

                    for to in quad_opinion[idx]:
                        start[to[0]] = 1
                        start[to[-1]] = 1
                    forward_answer_list.append([start, end])

                    # sentiment query
                    query = ["What", "sentiment", "given", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["and", "the",
                                                                                                        "opinion"]
                    for idy in range(len(quad_opinion[idx]) - 1):
                        to = quad_opinion[idx][idy]
                        query += text[to[0]:to[-1] + 1] + ["/"]
                    to = quad_opinion[idx][-1]
                    query += text[to[0]:to[-1] + 1] + ["?"]
                    sentiment_query_list.append(query)
                    sentiment_answer_list.append(quad_sentiment[idx])

                    # cate query--overall
                    query = ["What", "category", "does", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["belong", "to",
                                                                                                       "?"]

                    cate_query_list.append(query)
                    cate_answer_list.append(quad_cate[idx])

                    query = ["What", "general","category", "does", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["belong", "to",
                                                                                                       "?"]
                    category_query_list.append(query)
                    category_answer_list.append(quad_category[idx])

                    query = ["What", "attribute","of","the","category"]+ [category_list[quad_category[idx]]] +["does", "the", "aspect"] + text[ta[0]:ta[-1] + 1] + ["belong", "to",
                                                                                                       "?"]
                    attribute_query_list.append(query)
                    attribute_answer_list.append(quad_attribute[idx])



                    # aspect query
                for idx in range(len(dual_opinion)):
                    ta = dual_opinion[idx]
                    # opinion query
                    query = ["What", "aspect", "does", "the", "opinion"] + text[ta[0]:ta[-1] + 1] + ["describe", "?"]
                    backward_query_list.append(query)
                    start = [0] * len(text)
                    end = [0] * len(text)
                    for to in dual_aspect[idx]:
                        start[to[0]] = 1
                        end[to[-1]] = 1
                    backward_answer_list.append([start, end])


                # print(text)
                temp_sample = dual_sample(lines[k], text,context, forward_query_list, forward_answer_list, backward_query_list,
                                          backward_answer_list, sentiment_query_list, sentiment_answer_list, cate_query_list, cate_answer_list,
                                          category_query_list,category_answer_list,attribute_query_list,attribute_answer_list)
                sample_list.append(temp_sample)

            file = open(home_path + dataset_name + "/" + dataset_type + "_pair.pkl", "wb")
            pickle.dump(quad_list_save,file)
            file.close()
            torch.save(sample_list, output_path)