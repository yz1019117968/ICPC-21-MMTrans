#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:4/12/2020 1:06 PM
# contact: zhyang8-c@my.cityu.edu.hk

import pickle as pkl
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if __package__ is None:
    from Dictionary import Utils
else:
    import Utils




def create_dicts(dataset, sub_data_folder, num_words=30000):
    """
    create dictionaries for seqs and comms.
    :param dataset: dataset_train_val_test.pkl
    :param sub_data_folder: each model has his onw sub_data_folder
    :param num_words: max num of words in the dictionary
    :return:
    """
    sbts, nodes, _, comms = Utils.split_sbts_nodes_comms(dataset['train'])
    print("sbt[0]: ", sbts[0])
    print("nodes[0]: ", nodes[0])
    print("comms[0]: ", comms[0])
    Utils.create_dict(sub_data_folder, "sbts", sbts, num_words)
    Utils.create_dict(sub_data_folder, "nodes", nodes, num_words)
    Utils.create_dict(sub_data_folder, "comms", comms, num_words)


def generate_sequences(dataset, sub_data_folder):
    for type in ['train', 'val', 'test']:
        sbts, nodes, edges, comms = Utils.split_sbts_nodes_comms(dataset[type])
        sbts = Utils.text_to_idx(sub_data_folder, "sbts", sbts)
        nodes = Utils.text_to_idx(sub_data_folder, "nodes", nodes)
        comms = Utils.text_to_idx(sub_data_folder, "comms", comms)
        with open("../datasets/{}/tokens_idx/sbts_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(sbts, fw)
        with open("../datasets/{}/tokens_idx/nodes_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(nodes, fw)
        with open("../datasets/{}/tokens_idx/edges_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(edges, fw)
        with open("../datasets/{}/tokens_idx/comms_{}.pkl".format(sub_data_folder, type),"wb") as fw:
            pkl.dump(comms, fw)

def get_dicts(sub_data_folder):
    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    with open(parent_path + "/datasets/{}/dictionaries/sbts_dic.pkl".format(sub_data_folder), "rb") as fr:
        sbts_dic = pkl.load(fr)
    with open(parent_path + "/datasets/{}/dictionaries/nodes_dic.pkl".format(sub_data_folder), "rb") as fr:
        nodes_dic = pkl.load(fr)
    with open(parent_path + "/datasets/{}/dictionaries/comms_dic.pkl".format(sub_data_folder), "rb") as fr:
        comms_dic = pkl.load(fr)
    return sbts_dic, nodes_dic, comms_dic


if __name__ == "__main__":
    SUB_DATA_FOLDER = "smart_contracts/comms_4_20"
    with open("../datasets/{}/dataset_train_val_test_uniq.pkl".format(SUB_DATA_FOLDER), "rb") as fr:
        dataset = pkl.load(fr)
    # create_dicts(dataset, SUB_DATA_FOLDER)
    # generate_sequences(dataset, SUB_DATA_FOLDER)
    sbts_dic, nodes_dic, comms_dic = get_dicts(SUB_DATA_FOLDER)
    print(len(sbts_dic.word_index))
    print(len(nodes_dic.word_index))
    print(len(comms_dic.word_index))


