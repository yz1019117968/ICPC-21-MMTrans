#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:6/12/2020 10:01 AM
# contact: zhyang8-c@my.cityu.edu.hk

import re
import numpy as np
import pickle as pkl


def re_0002(i):
    # split camel case & snake case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0].lower(), tmp[1].lower())
    else:
        return ' '.format(tmp)
# first regex for removing special characters, the second for camelCase and snake_case
re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

re_opt = re.compile(r'\+|-|\*|/|\*\*|\+\+|--|%|<<|>>|&&|\|\||&|\|\^|<|>|<=|>=|==|!=|=|\|=|^=|&=|<<=|>>=|\+=|-=|\*=|/=|%=|:=|~|=:|_')





if __name__ == "__main__":
    # d0 = [123, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # d1 = ["123", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
    # d2 = ["abc fsdgsetrfd", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    # index = [i for i in range(len(d0))]
    # np.random.seed(123)
    # np.random.shuffle(index)
    # d0_train, d0_val, d0_test = split_dataset(d0, 0.1, 0.1, index)
    # d1_train, d1_val, d1_test = split_dataset(d1, 0.1, 0.1, index)
    # d2_train, d2_val, d2_test = split_dataset(d2, 0.1, 0.1, index)
    pass

