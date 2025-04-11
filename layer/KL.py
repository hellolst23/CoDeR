import numpy as np
def kl_div(p_dis, q_dis, alpha=0.01):
    KL_res = 0
    for index, p_value in enumerate(p_dis):
        if p_value < 1e-5:
            continue
        q_value = (1-alpha) * q_dis[index] + alpha * p_dis[index]
        KL_res += p_value*np.log(p_value/(q_value+1e-10))
    return KL_res

def user_kl_score(training_set, add_set, item_category):

    items_list = training_set + add_set
    items_len = len(items_list)
    training_set_1 = items_list[:items_len//2]
    training_set_2 = items_list[items_len//2:]
    
    training_set_1_dis = [0] * len(item_category)
    for itemID in training_set_1:
        categories = item_category[itemID]
        for cate in categories:
            training_set_1_dis[cate] += round(1.0/len(categories), 4)
    training_set_1_dis = [x/len(training_set_1) for x in training_set_1_dis]

    training_set_2_dis = [0] * len(item_category)
    for itemID in training_set_2:
        categories = item_category[itemID]
        for cate in categories:
            training_set_2_dis[cate] += round(1.0/len(categories), 4)
    training_set_2_dis = [x/len(training_set_2) for x in training_set_2_dis]
    
    kl_res_1 = kl_div(training_set_1_dis, training_set_2_dis)
    kl_res_2 = kl_div(training_set_2_dis, training_set_1_dis)
    
    return kl_res_1+kl_res_2

def get_kl_score(training_set_index, add_set_index, item_category):
    kl_score_set = np.zeros(len(training_set_index), dtype=float)
    for userID in range(len(training_set_index)):
        training_set = []
        for i in training_set_index[userID]:
            if i != 0:
                training_set.append(i)
        add_set = add_set_index[userID]
        kl_score = user_kl_score(training_set, [add_set], item_category)

        kl_score_set[userID] = kl_score

    avarage_kl_score = np.mean(kl_score_set)
    return kl_score_set, avarage_kl_score
