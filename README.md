### ICPC'21 MMTrans

#### A General Introduction for the Whole Framework.

- data_process: data pre-process.
- Dictionary: vocabulary related code.
- modules: model related code.
- DataOutput.py:  input data pipeline.
- Train.py: train the model.
- Evaluation.py: evaluate the model.
- EvaluationMetrics.py: automated evaluation metrics adopted in the experiment.
- Configs.py: model hyper-parameters setup.

#### Data and Trained Model

- Dataset is Available Here: [dataset](https://zenodo.org/record/4587089#.YEMmWugzYuU), put the datasets folder under the root directory.

- Trained Models: [Models](https://drive.google.com/drive/folders/1VkyISadwA8tp43xhJglqXIoSX3tT33k8?usp=sharing), put each of the checkpoint folder under the root directory, 
and the program will automatically load the latest model. (The number behind them is the head number setup of each experiment)

#### Other useful tools

- The SBT sequences and Graphs (xml format) generation tools are provided [here](https://github.com/yz1019117968/SC_tokenization).

- The model is implemented by [TensorFlow 2.3](https://www.tensorflow.org/tutorials/text/transformer?hl=zh-cn) based on the Transformer tutorial.  

#### Welcome to Cite!  
- If you find this paper or related tools useful and would like to cite it, the following would be appropriate:
```
@misc{yang2021multimodal,
      title={A Multi-Modal Transformer-based Code Summarization Approach for Smart Contracts}, 
      author={Zhen Yang and Jacky Keung and Xiao Yu and Xiaodong Gu and Zhengyuan Wei and Xiaoxue Ma and Miao Zhang},
      year={2021},
      eprint={2103.07164},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

  
