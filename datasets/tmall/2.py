# -*- coding: UTF-8 -*-
"""
@author:lifx1
@file:2.py
@time:2022/02/21
"""
import pickle

item_dict = pickle.load(open('new_item_dict.pkl', 'rb'))
catalog_dict = pickle.load(open('catalog_dict_dict.pkl', 'rb'))

session_train = pickle.load(open('../tmall_minoccur30_0321/session_train.pkl', 'rb'))
session_test = pickle.load(open('../tmall_minoccur30_0321/session_test.pkl', 'rb'))
# item_catgy = pickle.load(open('../tmall_minoccur30_0321/item_catgy.pkl', 'rb'))


train_session = []
for r in session_train:
    item_list = r[3]
    category_list = r[4]
    target_item = r[5]
    target_category = r[6]
    train_session.append(
        r[:3] + [[item_dict[x] for x in item_list], [catalog_dict[x] for x in category_list], item_dict[target_item],
                 catalog_dict[target_category]])

pickle.dump(train_session, open('session_train.pkl', 'wb'))

test_session = []
for r in session_test:
    item_list = r[3]
    category_list = r[4]
    target_item = r[5]
    target_category = r[6]
    test_session.append(
        r[:3] + [[item_dict[x] for x in item_list], [catalog_dict[x] for x in category_list], item_dict[target_item],
                 catalog_dict[target_category]])

pickle.dump(test_session, open('session_test.pkl', 'wb'))

item_catalog_dict = dict()

for r in train_session:
    item_list = r[3]
    category_list = r[4]
    target_item = r[5]
    target_category = r[6]

    for item, catalog in zip(item_list, category_list):
        # if item in item_catalog_dict:
            # if item_catalog_dict[item] != catalog:
            #     print(item_catalog_dict[item])
            #     print(item_list)
            #     print(category_list)
            #     print(item, item_catalog_dict[item])
            #     raise TypeError('123')
        item_catalog_dict[item] = catalog

    # if target_item in item_catalog_dict:
    #     assert item_catalog_dict[target_item] == target_category
    item_catalog_dict[target_item] = target_category

for r in test_session:
    item_list = r[3]
    category_list = r[4]
    target_item = r[5]
    target_category = r[6]

    for item, catalog in zip(item_list, category_list):
        # if item in item_catalog_dict:
            # if item_catalog_dict[item] != catalog:
            #     print(item_catalog_dict[item])
            #     print(item_list)
            #     print(category_list)
            #     print(item, item_catalog_dict[item])
            #     raise TypeError('123')
        item_catalog_dict[item] = catalog

    # if target_item in item_catalog_dict:
    #     assert item_catalog_dict[target_item] == target_category
    item_catalog_dict[target_item] = target_category

item_catgy=[]
for k in sorted(list(item_catalog_dict.keys())):
    item_catgy.append(item_catalog_dict[k])

print(len(item_dict))
print(len(item_catalog_dict))
print(len(item_catgy))
pickle.dump(item_catgy, open('item_catgy.pkl', 'wb'))