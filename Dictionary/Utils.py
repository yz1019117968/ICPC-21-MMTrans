#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:4/12/2020 1:16 PM
# contact: zhyang8-c@my.cityu.edu.hk

import pickle as pkl
from tensorflow.keras.preprocessing.text import Tokenizer



def split_sbts_nodes_comms(dataset):
    """
    Returns processed database

    :param dataset: list of sentence pairs
    :return: list of paralel data e.g.
    (['first source sentence', 'second', ...], ['first target sentence', 'second', ...])
    """
    sbts = []
    nodes = []
    comms = []
    edges = []
    for sbt, node, edge, comm in dataset:
        sbts.append(sbt)
        nodes.append(node)
        comms.append(comm)
        edges.append(edge)
    return sbts, nodes, edges, comms


def create_dict(sub_data_folder, type, texts, num_words=30000):
    """
    :param sub_data_folder: each model has his onw sub_data_folder
    :param type: "seqs" or "comms".
    :param texts: a list of string, where each string is a sentence with space as separation.
    :param num_words: the vocabulary size.
    :return:
    """
    tk = Tokenizer(num_words=num_words+1, lower=False, split=' ', char_level=False, oov_token="<UNK>", filters="")
    tk.fit_on_texts(texts)
    tk.word_index = {e:i for e,i in tk.word_index.items() if i <= num_words+1}
    with open("../datasets/{}/dictionaries/{}_dic.pkl".format(sub_data_folder, type), "wb") as fw:
        pkl.dump(tk,fw)

def add_start_end(total_num, seqs):
    new_seqs = []
    for seq in seqs:
        new_seqs.append([total_num+1] + seq + [total_num+2])
    return new_seqs


def text_to_idx(sub_data_folder, type, texts):
    """
    return an 2d array, each row represent a sentence with a max_len length.
    :param type: "seqs" or "comms".
    :param texts: a list of string, where each string is a sentence with space as separation.
    :param max_len: the max length of a sentence.
    :return:
    """
    with open("../datasets/{}/dictionaries/{}_dic.pkl".format(sub_data_folder, type), "rb") as fr:
        tk = pkl.load(fr)
    print(texts[0])
    seqs = tk.texts_to_sequences(texts)
    total_num = len(tk.word_index)
    print("Total_num: ", total_num)
    if type != "nodes":
        seqs = add_start_end(total_num, seqs)
    print(seqs[0])
    return seqs

