import os
import math
import numpy as np

def kl_div(p_dis, q_dis, alpha=0.01):
    KL_res = 0
    for index, p_value in enumerate(p_dis):
        if p_value < 1e-5:
            continue
        q_value = (1-alpha) * q_dis[index] + alpha * p_dis[index]
        KL_res += p_value*(np.log(p_value/q_value))
    return KL_res

def user_kl_score(training_set, validation_set, item_category, category_list):
    items_list = training_set + validation_set
    items_len = len(items_list)
    training_set_1 = items_list[:items_len//2]
    training_set_2 = items_list[items_len//2:]
    
    training_set_1_dis = [0] * len(category_list)
    for itemID in training_set_1:
        categories = item_category[itemID]
        for cate in categories:
            training_set_1_dis[cate] += round(1.0/len(categories), 4)
    training_set_1_dis = [x/len(training_set_1) for x in training_set_1_dis]

    training_set_2_dis = [0] * len(category_list)
    for itemID in training_set_2:
        categories = item_category[itemID]
        for cate in categories:
            training_set_2_dis[cate] += round(1.0/len(categories), 4)
    training_set_2_dis = [x/len(training_set_2) for x in training_set_2_dis]
    
    kl_res_1 = kl_div(training_set_1_dis, training_set_2_dis)
    kl_res_2 = kl_div(training_set_2_dis, training_set_1_dis)
    
    return kl_res_1+kl_res_2


def get_kl_score(training_set_index, validation_set_index, item_category, category_list):
    """
    Arg:
    :param training_set_index: test example
    :param validation_set_index:
    :param item_category:
    :param category_list:

    :return: kl_score_set, avarage_kl_score, max_kl_score, min_kl_score

    """

    max_kl_score = 0
    min_kl_score = 999
    kl_score_set = np.zeros(len(training_set_index), dtype=float)
    for userID in range(len(training_set_index)):
        training_set = []
        for i in training_set_index[userID]:
            if i != 0:
                training_set.append(i)
        # print(training_set)
        validation_set = validation_set_index[userID]
        # print([validation_set])
        kl_score = user_kl_score(training_set, [validation_set], item_category, category_list)

        kl_score_set[userID] = kl_score
        if kl_score > max_kl_score:
            max_kl_score = kl_score
        if kl_score < min_kl_score:
            min_kl_score = kl_score

    avarage_kl_score = np.mean(kl_score_set)
    return kl_score_set, avarage_kl_score, max_kl_score, min_kl_score

# if __name__ == "__main__":
#     # user_kl_score = {}
#     train_set = [1,2,3,5,6]
#     valid_set = [3,4]
#     item_category = {1:[1,5,6] , 2:[5], 3:[2,5], 4:[2,6], 5:[3], 6:[1,6]}

#     category_list = ['Animation',"Children's" ,'Comedy' ,'Adventure' ,'Fantasy' ,'Romance', 'Drama'
#                     'Action', 'Crime' ,'Thriller' ,'Horror', 'Sci-Fi' ,'Documentary','War'
#                     'Musical','Mystery','Film-Noir','Western'] 

#     max_kl_score = 0
#     min_kl_score = 999

#     kl_score = user_kl_score(train_set, valid_set, item_category, category_list)

#     # user_kl_score[userID] = kl_score
#     # if kl_score > max_kl_score:
#     #     max_kl_score = kl_score
#     # if kl_score < min_kl_score:
#     #     min_kl_score = kl_score

#     # for userID in FM_user_item_score:
#         # train_set = train_dict[userID]
#         # valid_set = valid_dict[userID]
#         # kl_score = util.user_kl_score(train_set, valid_set, item_category, category_list)
#         # user_kl_score[userID]=kl_score
#         # if kl_score>max_kl_score:
#         #     max_kl_score = kl_score
#         # if kl_score<min_kl_score:
#         #     min_kl_score = kl_score
    
#     # print(min_kl_score)
#     # print(max_kl_score)

#     training_set_index = [[1,2,3,5,6,0,0,0,0,0,0],[1,2,4,3,1,0,0,0,0,0,0],[5,6,2,3,4,0,0,0,0,0,0]]
#     validation_set_index = [2,3,4] 
#     # print([validation_set_index[0]])
#     user_kl_score, avarage_kl_score, max_kl_score, min_kl_score = get_kl_score(training_set_index, validation_set_index, item_category, category_list)
#     print(user_kl_score)
#     print(avarage_kl_score)
