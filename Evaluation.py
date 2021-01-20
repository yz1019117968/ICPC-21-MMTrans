#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:24/12/2020 2:42 PM
# contact: zhyang8-c@my.cityu.edu.hk

from modules.MMTrans import MMTrans
from modules.TransformerUtils import TransformerUtils as utils
import tensorflow as tf
from Dictionary.Dictionary import get_dicts
from DataOutput import test_data_prepare, MAX_LENGTH_COMM
from EvaluationMetrics import EvaluationMetrics
from modules.CustomSchedule import CustomSchedule
from Configs import Eval_args
import numpy as np

class Evaluation:
    def __init__(self, sub_data_folder, transformer_args):
        self.sub_data_folder = sub_data_folder
        self.transformer = MMTrans(**transformer_args)
        self.learning_rate = CustomSchedule(transformer_args['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def test_step(self, srcs_test, nodes_test, edges_test, comms_test, eva_metrics):
        current_batch_size = comms_test.shape[0]
        tar_reals = comms_test[:, 1:]
        _, _, comms_dic = get_dicts(self.sub_data_folder)
        # add start tag as the first input for the decoder
        decoder_input = [len(comms_dic.word_index)+1] * current_batch_size
        outputs = tf.expand_dims(decoder_input, 1)
        for i in range(MAX_LENGTH_COMM-1):
            src_padding_mask, node_padding_mask, look_ahead_mask = utils.create_masks(srcs_test, nodes_test, outputs)
            predictions, attention_weights, _, _ = self.transformer(srcs_test, nodes_test, edges_test, outputs, False,
                         src_padding_mask, node_padding_mask, look_ahead_mask)

            predictions = predictions[:, -1:, :]
            predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            outputs = tf.concat([outputs, predicted_ids], axis=-1)
            count_stopped = 0
            for output in outputs:
                if len(comms_dic.word_index) + 2 in output.numpy().tolist():
                    count_stopped += 1
            if count_stopped == current_batch_size:
                break
        candidates = EvaluationMetrics.remove_pad(outputs.numpy().tolist(), len(comms_dic.word_index)+2, "candidates")
        refs = EvaluationMetrics.remove_pad(tar_reals.numpy().tolist(), len(comms_dic.word_index)+2, "references")
        for ref, candi in zip(refs, candidates):
            eva_metrics['sentence_bleu'].append(EvaluationMetrics.smoothing1_sentence_bleu(ref, candi))
            eva_metrics['rouge'].append(EvaluationMetrics.rouge(ref, candi))
            eva_metrics['meteor'].append(EvaluationMetrics.meteor(ref, candi))
        eva_metrics['corpus_bleu'].append(EvaluationMetrics.smoothing1_corpus_bleu(refs, candidates))


    def evaluate(self):
        test_set = test_data_prepare(self.sub_data_folder)
        eva_metrics = {'sentence_bleu': [], 'corpus_bleu': [], 'rouge': [], 'meteor': []}
        print("Validating ...")

        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        checkpoint_path = "./checkpoints_4/" + self.sub_data_folder
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
        # 如果检查点存在，则恢复最新的检查点。
        if ckpt_manager.latest_checkpoint:
            print("ckpt: ", ckpt_manager.latest_checkpoint)
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        for batch, (srcs_test, nodes_test, edges_test, comms_test) in enumerate(test_set):
            self.test_step(srcs_test, nodes_test, edges_test, comms_test, eva_metrics)
            if batch % 50 == 0:
                print("Validating the {} batch".format(batch))

        print("Validation Completed!")
        for metric in eva_metrics.keys():
            print("{}: {:.4f}".format(metric, np.mean(eva_metrics[metric])))

    def sequence_to_text(self, reverse_word_map, list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map[int(letter)] for letter in list_of_indices]
        return words

    def evaluate_example(self):
        content = ""
        test_set = test_data_prepare(self.sub_data_folder)
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        checkpoint_path = "./checkpoints_4/" + self.sub_data_folder
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
        # 如果检查点存在，则恢复最新的检查点。
        if ckpt_manager.latest_checkpoint:
            print("ckpt: ", ckpt_manager.latest_checkpoint)
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        else:
            print("cannot load latest model!!")
        for batch, (srcs_test, nodes_test, edges_test, comms_test) in enumerate(test_set):
            current_batch_size = comms_test.shape[0]
            tar_reals = comms_test[:, 1:]
            srcs_dic, nodes_dic, comms_dic = get_dicts(self.sub_data_folder)
            # add start tag as the first input for the decoder
            decoder_input = [len(comms_dic.word_index)+1] * current_batch_size
            outputs = tf.expand_dims(decoder_input, 1)
            for i in range(MAX_LENGTH_COMM-1):
                src_padding_mask, node_padding_mask, look_ahead_mask = utils.create_masks(srcs_test, nodes_test, outputs)
                predictions, attention_weights = self.transformer(srcs_test, nodes_test, edges_test, outputs, False,
                             src_padding_mask, node_padding_mask, look_ahead_mask)

                predictions = predictions[:, -1:, :]
                predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

                outputs = tf.concat([outputs, predicted_ids], axis=-1)
                count_stopped = 0
                for output in outputs:
                    if len(comms_dic.word_index) + 2 in output.numpy().tolist():
                        count_stopped += 1
                if count_stopped == current_batch_size:
                    break
            candidates = EvaluationMetrics.remove_pad(outputs.numpy().tolist(), len(comms_dic.word_index)+2, "candidates")
            refs = EvaluationMetrics.remove_pad(tar_reals.numpy().tolist(), len(comms_dic.word_index)+2, "references")

            reverse_word_map_comms = dict(map(reversed, comms_dic.word_index.items()))
            reverse_word_map_srcs = dict(map(reversed, srcs_dic.word_index.items()))

            content += (str(batch) + ":\n")
            content += ("src: " + " ".join(self.sequence_to_text(reverse_word_map_srcs, srcs_test.numpy()[0][1:-1])) + "\n")
            content += ("candidate: " + " ".join(self.sequence_to_text(reverse_word_map_comms, candidates[0])) + "\n")
            content += ("ref: " + " ".join(self.sequence_to_text(reverse_word_map_comms, refs[0][0])) + "\n\n")
            print(batch)
        with open("./final results/all_results_wo_gru.txt", "w", encoding="utf-8") as fw:
            fw.write(content)

    def evaluate_attn(self, idx):
        """
        return the attn of the test case in designated idx.
        :param idx:
        :return:
        """
        test_set = test_data_prepare(self.sub_data_folder)
        ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                   optimizer=self.optimizer)
        checkpoint_path = "./checkpoints_4/" + self.sub_data_folder
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
        # 如果检查点存在，则恢复最新的检查点。
        if ckpt_manager.latest_checkpoint:
            print("ckpt: ", ckpt_manager.latest_checkpoint)
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        else:
            print("cannot load latest model!!")
        for batch, (srcs_test, nodes_test, edges_test, comms_test) in enumerate(test_set):

            if idx == batch:
                current_batch_size = comms_test.shape[0]
                srcs_dic, nodes_dic, comms_dic = get_dicts(self.sub_data_folder)
                # add start tag as the first input for the decoder
                decoder_input = [len(comms_dic.word_index)+1] * current_batch_size
                outputs = tf.expand_dims(decoder_input, 1)
                # attn_graph = -1
                # attn_sbt = -1

                for i in range(MAX_LENGTH_COMM-1):
                    src_padding_mask, node_padding_mask, look_ahead_mask = utils.create_masks(srcs_test, nodes_test, outputs)
                    predictions, attention_weights, sbt_attn, graph_attn = self.transformer(srcs_test, nodes_test, edges_test, outputs, False,
                                 src_padding_mask, node_padding_mask, look_ahead_mask)
                    # for v in self.transformer.trainable_variables:
                    #     if v.name == "transformer_yz/encoder_graph/graph_embed/embeddings:0":
                    #         v[0]
                    predictions = predictions[:, -1:, :]
                    predicted_ids = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

                    outputs = tf.concat([outputs, predicted_ids], axis=-1)


                    count_stopped = 0
                    for output in outputs:
                        if len(comms_dic.word_index) + 2 in output.numpy().tolist():
                            count_stopped += 1
                    if count_stopped == current_batch_size:
                        break
                print(edges_test)
                candidates = EvaluationMetrics.remove_pad(outputs.numpy().tolist(), len(comms_dic.word_index)+2, "candidates")
                reverse_word_map_comms = dict(map(reversed, comms_dic.word_index.items()))
                reverse_word_map_nodes = dict(map(reversed, nodes_dic.word_index.items()))
                reverse_word_map_srcs = dict(map(reversed, srcs_dic.word_index.items()))
                candidates = self.sequence_to_text(reverse_word_map_comms, candidates[0])
                _nodes_test = self.sequence_to_text(reverse_word_map_nodes, nodes_test.numpy()[0])
                _srcs_test = self.sequence_to_text(reverse_word_map_srcs, srcs_test.numpy()[0][1:-1])
                print(_nodes_test)
                # self.plot_single(sbt_attn, _srcs_test, _srcs_test, "srcs")
                # self.plot_attention_weights(attention_weights['decoder_layer1_graph_comm'], _nodes_test, candidates, "nodes")
                # self.plot_attention_weights(attention_weights['decoder_layer1_sbt_comm'], _srcs_test, candidates, "srcs")
                self.plot_std_attn(attention_weights['decoder_layer1_graph_comm'], _nodes_test, candidates, "nodes")
                self.plot_std_attn(attention_weights['decoder_layer1_sbt_comm'], _srcs_test, candidates, "srcs")


    def standard_attn(self, mat):
        mat = mat.transpose((1,0,2))
        for row in range(mat.shape[0]):
            row_max = np.max(mat[row])
            row_min = np.min(mat[row])
            mat[row] = (mat[row] - row_min) / (row_max - row_min)
        return mat

    def plot_attention_weights(self, attention, sentence, result, inp):
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(15, 60))

        attention = tf.squeeze(attention, axis=0).numpy()
        for head in range(attention.shape[0]):

            ax = fig.add_subplot(4, 1, head+1)

            # 画出注意力权重
            ax.matshow(attention[head], cmap='viridis')
            fontdict = {'fontsize': 8}
            if inp == "srcs":
                ax.set_xticks(range(len(sentence)+2))
            elif inp == "comms":
                ax.set_xticks(range(len(sentence)+1))
            else:
                ax.set_xticks(range(len(sentence)))
            ax.set_yticks(range(len(result)+1))

            ax.set_ylim(len(result), -0.5)
            if head == 0:
                if inp == "srcs":
                    ax.set_xticklabels(['<start>'] + sentence + ['<end>'],
                        fontdict=fontdict, rotation=90)
                elif inp == "comms":
                    ax.set_xticklabels(['<start>'] + sentence,
                        fontdict=fontdict, rotation=90)
                else:
                    ax.set_xticklabels(sentence,
                        fontdict=fontdict, rotation=90)
            else:
                if inp == "srcs":
                    ax.set_xticklabels(np.arange(len(sentence) + 2), fontdict=fontdict)
                elif inp == "comms":
                    ax.set_xticklabels(np.arange(len(sentence) + 1), fontdict=fontdict)
                else:
                    ax.set_xticklabels(np.arange(len(sentence)), fontdict=fontdict)
            ax.set_yticklabels(["<start>"] + result, fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head+1))


        # plt.tight_layout()
        plt.show()

    def plot_std_attn(self, attention, sen_x, sen_y, inp):
        from matplotlib import pyplot as plt
        sen_y = ['<start>'] + sen_y
        attention = tf.squeeze(attention, axis=0).numpy()
        attention = self.standard_attn(attention)
        for row in range(attention.shape[0]):
            plt.matshow(attention[row], cmap='viridis')
            plt.colorbar()
            if inp == "srcs":
                plt.xticks(np.arange(len(sen_x) + 2), ['<start>'] + sen_x + ['<end>'],
                    fontsize=8, rotation=90)
            elif inp == "comms":
                plt.xticks(np.arange(len(sen_x) + 1), ['<start>'] + sen_x,
                    fontsize=8, rotation=90)
            else:
                plt.xticks(np.arange(len(sen_x)), tuple(sen_x),
                    fontsize=8, rotation=90)
            plt.ylim(attention.shape[1] - 0.5)
            plt.yticks(range(attention.shape[1]))
            y = sen_y[0:row+2]
            y.reverse()
            plt.ylabel(" ".join(y))
            plt.show()



if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # TODO: Remember to change parameters when implement  the evaluation
    evaluation = Evaluation(**Eval_args)
    evaluation.evaluate()
    # evaluation.evaluate_example()
    # evaluation.evaluate_attn(1052)


