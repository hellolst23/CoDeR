# -*- coding: UTF-8 -*-
"""
@file:1.py
"""
import pickle

train64 = pickle.load(open('../tmall_minoccur30_0321/train.txt', 'rb'))
test64 = pickle.load(open('../tmall_minoccur30_0321/test.txt', 'rb'))
train_c = pickle.load(open('../tmall_minoccur30_0321/train_c.txt', 'rb'))
test_c = pickle.load(open('../tmall_minoccur30_0321/test_c.txt', 'rb'))

catalog_set = set()
for catalog_list in train_c[0]:
    for catalog in catalog_list:
        catalog_set.add(catalog)

for catalog in train_c[1]:
    catalog_set.add(catalog)

for catalog_list in test_c[0]:
    for catalog in catalog_list:
        catalog_set.add(catalog)

for catalog in test_c[1]:
    catalog_set.add(catalog)

catalog_list = sorted(list(catalog_set))
catalog_dict = dict()
for i in range(1, len(catalog_set) + 1):
    catalog_dict[catalog_list[i - 1]] = i

train64_x = train64[0]
train64_y = train64[1]

test64_x = test64[0]
test64_y = test64[1]
train_pos = list()
test_pos = list()

# renumber of item and generate pos
item_set = set()

for items in train64_x:
    pos = list()
    for id_ in range(len(items)):
        item_set.add(items[id_])
        pos.append(id_ + 1)
    pos.append(len(items) + 1)
    train_pos.append(pos)

for item in train64_y:
    item_set.add(item)

for items in test64_x:
    pos = []
    for id_ in range(len(items)):
        item_set.add(items[id_])
        pos.append(id_ + 1)
    pos.append(len(items) + 1)
    test_pos.append(pos)

for item in test64_y:
    item_set.add(item)

item_list = sorted(list(item_set))

print(len(item_list))
item_dict = dict()
for i in range(1, len(item_set) + 1):
    item = item_list[i - 1]
    item_dict[item] = i

print(max(list(set(test64_y))))
print(max(list(set(train64_y))))

train64_x_new = list()
train64_y_new = list()

test64_x_new = list()
test64_y_new = list()

train_x_catalog = list()
train_y_catalog = list()

test_x_catalog = list()
test_y_catalog = list()

for index1, items in enumerate(train64_x):
    new_list = []
    catalog_list = []
    for index2, item in enumerate(items):
        new_list.append(item_dict[item])
        catalog_list.append(catalog_dict[train_c[0][index1][index2]])
    train64_x_new.append(new_list)
    train_x_catalog.append(catalog_list)

for index, item in enumerate(train64_y):
    train64_y_new.append(item_dict[item])
    train_y_catalog.append(catalog_dict[train_c[1][index]])

for index1, items in enumerate(test64_x):
    new_list = []
    catalog_list = []
    for index2, item in enumerate(items):
        new_list.append(item_dict[item])
        catalog_list.append(catalog_dict[test_c[0][index1][index2]])
    test64_x_new.append(new_list)
    test_x_catalog.append(catalog_list)

for index, item in enumerate(test64_y):
    test64_y_new.append(item_dict[item])
    test_y_catalog.append(catalog_dict[test_c[1][index]])

train64_x = train64_x_new
train64_y = train64_y_new
test64_x = test64_x_new
test64_y = test64_y_new

pickle.dump([train64_x, train64_y], open('train.txt', 'wb'))
pickle.dump([test64_x, test64_y], open('test.txt', 'wb'))
pickle.dump([train_x_catalog, train_y_catalog], open('train_c.txt', 'wb'))
pickle.dump([test_x_catalog, test_y_catalog], open('test_c.txt', 'wb'))

pickle.dump(item_dict, open('new_item_dict.pkl', 'wb'))
pickle.dump(catalog_dict, open('catalog_dict_dict.pkl', 'wb'))
import json

json.dump({
    "num_of_item": len(item_dict),
    "num_of_category": len(catalog_dict),
}, open('session_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
