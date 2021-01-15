#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:17/11/2020 8:47 PM
# contact: zhyang8-c@my.cityu.edu.hk

from Dictionary.Dictionary import get_dicts
SUB_DATA_FOLDER = "smart_contracts/comms_4_20"
SBTS_DIC, NODES_DIC, COMMS_DIC = get_dicts(SUB_DATA_FOLDER)



Transformer_args = {
"num_layers": 1,
"d_model": 256,
"dff": 512,
"num_heads": 4,
"srcs_vocab_size": len(SBTS_DIC.word_index) + 3,
"graphs_vocab_size": len(NODES_DIC.word_index) + 1,
"asthop": 2,
"comms_vocab_size": len(COMMS_DIC.word_index) + 3,
"pe_srcs":len(SBTS_DIC.word_index) + 3,
"pe_graphs":len(NODES_DIC.word_index) + 1,
"pe_comms":len(COMMS_DIC.word_index) + 3,
"rate": 0.2
}

Train_args = {
    "epochs": 50,
    "sub_data_folder": SUB_DATA_FOLDER,
    "max_keep": 10,
    "patience": 5,
    "transformer_args": Transformer_args
}

Eval_args = {
    "sub_data_folder": SUB_DATA_FOLDER,
    "transformer_args": Transformer_args
}

