#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:13/11/2020 10:00 AM
# contact: zhyang8-c@my.cityu.edu.hk
import os
import re

from data_process.utils import re_0001_, re_0002, re_opt
from data_process.xml_to_graph import xml_graph
import pickle as pkl
import random

def process_sbt_seq(contract_folder, seq_file):
    """
    process camelCase and snake_case names
    :param contract_folder:
    :param seq_file:
    :return: a string of seq
    """
    with open("../contracts/contracts_seqs_sbt/{}/{}".format(contract_folder, seq_file), "r", encoding="utf-8") as fr:
    # with open("../contracts/test_sbt.txt", "r", encoding="utf-8") as fr:
        seq = fr.readline()
        tokens = seq.split(" ")
        new_tokens = []
        for token in tokens:
            if token.strip().startswith("SimpleName") and token.find("#"):
                type = token.strip().split("#")[0].lower()
                new_tokens.append(type)
                value = token.strip().split("#")[1]
                if re_opt.fullmatch(value) is None:
                    value = re_0001_.sub(re_0002, value).strip().lower()
                new_tokens.append(value)
            else:
                if token != "$NUM$" and token != "$STR$" and token != "$ADDR$":
                    token = token.lower()
                else:
                    token = "「" + token[1:-1] + "」"
                new_tokens.append(token)
        return " ".join(new_tokens)


def process_sbts_graphs_comms(min_comm_len, max_comm_len):
    comm_contracts_path = "../contracts/comments_v11162020"
    contracts = os.listdir(comm_contracts_path)
    total_contracts = len(contracts)

    dataset = []
    for idx, contract in enumerate(contracts):
        comm_files_path = comm_contracts_path + "/" + contract
        files = os.listdir(comm_files_path)

        for file in files:
            with open(comm_files_path + "/" + file, "r", encoding="utf-8") as fcr:
                comm = fcr.readline()
            comm_tokens = comm.split(" ")
            if len(comm_tokens) <= max_comm_len and len(comm_tokens) >= min_comm_len:
                sbt_seq = process_sbt_seq(contract, file.replace("_comm", ""))
                nodes, edges = xml_graph(contract, file.replace("_comm", ""))
                dataset.append((sbt_seq, nodes, edges, comm))
        print(str(idx) + "/" + str(total_contracts) + " finished!")
    with open("../datasets/smart_contracts/comms_{}_{}/dataset.pkl".format(min_comm_len, max_comm_len), "wb") as fw:
        pkl.dump(dataset, fw)


def split_dataset(dataset_name, test_prob, val_prob, min_comm_len = 4, max_comm_len = 20):

    with open("../datasets/smart_contracts/comms_{}_{}/{}.pkl".format(min_comm_len, max_comm_len, dataset_name), "rb") as fr:
        dataset = pkl.load(fr)
    print(dataset[0])
    random.Random(345).shuffle(dataset)
    total_length = len(dataset)
    val_num = int(total_length * test_prob)
    test_num = int(total_length * val_prob)
    test_set = dataset[0: test_num]
    val_set = dataset[test_num: test_num+val_num]
    train_set = dataset[test_num+val_num:]
    new_dataset = {'train':train_set, 'val': val_set, 'test': test_set}
    with open("../datasets/smart_contracts/comms_{}_{}/dataset_train_val_test.pkl".format(min_comm_len, max_comm_len), "wb") as fw:
        pkl.dump(new_dataset,fw)

def refine_dataset(min_comm_len = 4, max_comm_len = 20):
    with open("../datasets/smart_contracts/comms_{}_{}/dataset_train_val_test.pkl".format(min_comm_len, max_comm_len), "rb") as fr:
        dataset = pkl.load(fr)
    with open("../datasets/smart_contracts/comms_{}_{}/uniq_val_idics".format(min_comm_len, max_comm_len), "rb") as fr:
        val_list = pkl.load(fr)
    with open("../datasets/smart_contracts/comms_{}_{}/uniq_test_idics_x".format(min_comm_len, max_comm_len), "rb") as fr:
        test_list = pkl.load(fr)
    new_val_set = []
    for idx in val_list:
        new_val_set.append(dataset['val'][idx])
    new_test_set = []
    for idx in test_list:
        new_test_set.append(dataset['test'][idx])
    dataset.update({'val': new_val_set, 'test': new_test_set})
    print(dataset.keys())
    print(len(dataset['val']))
    with open("../datasets/smart_contracts/comms_{}_{}/dataset_train_val_test_uniq.pkl".format(min_comm_len, max_comm_len), "wb") as fw:
        pkl.dump(dataset, fw)


if __name__ == "__main__":
    # process_sbts_graphs_comms(4, 20)
    # split_dataset("dataset", 0.05, 0.05)
    # print(process_sbt_seq("test", "test"))
    # refine_dataset()

    with open("../datasets/smart_contracts/comms_{}_{}/dataset_train_val_test_uniq.pkl".format(4, 20), "rb") as fw:
        data = pkl.load(fw)

    for idx, (src_seq, nodes, edges, comm) in enumerate(data['test']):
        print(comm)
        if idx > 5:
            break







