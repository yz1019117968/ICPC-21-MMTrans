#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:19/11/2020 8:11 PM
# contact: zhyang8-c@my.cityu.edu.hk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import tensorflow as tf
import numpy as np
from rouge import Rouge

class EvaluationMetrics:

    @staticmethod
    def smoothing1_sentence_bleu(reference, candidate):
        chencherry = SmoothingFunction()
        return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    @staticmethod
    def smoothing1_corpus_bleu(references, candidates):
        chencherry = SmoothingFunction()
        return corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

    @staticmethod
    def rouge(reference, candidate):
        rouge = Rouge()
        return rouge.get_scores(" ".join(candidate), " ".join(reference[0]))[0]['rouge-l']['f']

    @staticmethod
    def meteor(reference, candidate):
        return meteor_score([" ".join(reference[0])], " ".join(candidate))

    @staticmethod
    def remove_pad(tokens_list, end_idx, type):
        outputs = []
        for tokens in tokens_list:
            output = []
            for token in tokens:
                if token == end_idx:
                    break
                else:
                    output.append(str(token))
            assert type == "candidates" or type == "references"
            if type == "candidates":
                outputs.append(output[1:])
            elif type == "references":
                outputs.append([output])
        return outputs

if __name__ == "__main__":
    candi = tf.constant([[1234, 12, 4, 5, 34, 1235, 0, 0], [1234, 22, 41, 35, 12, 1235, 0, 0], [1234, 34, 23, 22, 34, 123, 33, 23]])
    candi = candi.numpy().tolist()
    candi = EvaluationMetrics.remove_pad(candi, 1235, "candidates")
    # print(candi)
    refs = tf.constant([[12, 4, 5, 34, 1235, 0, 0, 0], [22, 41, 34, 12, 1235, 0, 0, 0], [34, 23, 22, 34, 123, 33, 23, 1235]])
    refs = EvaluationMetrics.remove_pad(refs.numpy().tolist(), 1235, "references")
    print("refs: ", refs)
    a = []
    for candidate, ref in zip(candi, refs):
        a.append(EvaluationMetrics.smoothing1_sentence_bleu(ref, candidate))
    print(np.mean(a))
    print(EvaluationMetrics.smoothing1_corpus_bleu(refs, candi))

