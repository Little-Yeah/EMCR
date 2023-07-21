# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import torch
import pickle




def make_standard(home_path, dataset_name, dataset_type):
    # read triple
    standard_list = []
    f = open(home_path + dataset_name + "/" + dataset_type + "_pair.pkl", "rb")
    triple_data = pickle.load(f)
    f.close()

    for triplet in triple_data:

        aspect_temp = []
        opinion_temp = []
        pair_temp = []
        triplet_temp = []
        categorypair_temp = []
        asp_pol_temp = []
        asp_cate_temp = []
        asp_category_temp = []
        asp_attribute_temp = []
        for temp_t in triplet:
            triplet_temp.append([temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1], temp_t[2], temp_t[4],temp_t[5]])

            ap = [temp_t[0][0], temp_t[0][-1], temp_t[2]]
            if ap not in asp_pol_temp:
                asp_pol_temp.append(ap)
            ac = [temp_t[0][0], temp_t[0][-1], temp_t[3]]
            if ac not in asp_cate_temp:
                asp_cate_temp.append(ac)
            a = [temp_t[0][0], temp_t[0][-1]]
            if a not in aspect_temp:
                aspect_temp.append(a)
            o = [temp_t[1][0], temp_t[1][-1]]
            if o not in opinion_temp:
                opinion_temp.append(o)
            p = [temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1]]
            if p not in pair_temp:
                pair_temp.append(p)
            
            acategory = [temp_t[0][0], temp_t[0][-1], temp_t[4]]
            if acategory not in asp_category_temp:
                asp_category_temp.append(acategory)
            aattribute = [temp_t[0][0], temp_t[0][-1], temp_t[5]]
            if aattribute not in asp_attribute_temp:
                asp_attribute_temp.append(aattribute)
            category_pair = [temp_t[4],temp_t[5]]
            if category_pair not in categorypair_temp:
                categorypair_temp.append(category_pair)

        # print(triplet_temp)


        standard_list.append({'asp_target': aspect_temp, 'opi_target': opinion_temp, 'asp_opi_target': pair_temp,
                     'asp_pol_target': asp_pol_temp, 'asp_cate_target':asp_cate_temp, 'asp_category_target':asp_category_temp,
                     'asp_attribute_target':asp_attribute_temp, 'triplet': triplet_temp,'category_pair':categorypair_temp})

    return standard_list


if __name__ == '__main__':
    home_path = "./data/original/"
    dataset_name_list = ['Laptop_ACOS']
    for dataset_name in dataset_name_list:
        output_path = "./data/preprocess/" + dataset_name + "_standard.pt"
        dev_standard = make_standard(home_path, dataset_name, 'dev')
        test_standard = make_standard(home_path, dataset_name, 'test')
        torch.save({'dev': dev_standard, 'test': test_standard}, output_path)
