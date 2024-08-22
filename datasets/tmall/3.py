# -*- coding: UTF-8 -*-
"""
@file:3.py
"""
import pickle

train64 = pickle.load(open('../tmall_minoccur30_0321/train.txt', 'rb'))
test64 = pickle.load(open('../tmall_minoccur30_0321/test.txt', 'rb'))
train_c = pickle.load(open('../tmall_minoccur30_0321/train_c.txt', 'rb'))
test_c = pickle.load(open('../tmall_minoccur30_0321/test_c.txt', 'rb'))
item_catgy = pickle.load(open('../tmall_minoccur30_0321/item_catgy.pkl', 'rb'))

print(train64[0][0])
print(train_c[0][0])

for index in range(len(train64[0][0])):
    print(train64[0][0][index], train_c[0][0][index], item_catgy[train64[0][0][index] - 1])
