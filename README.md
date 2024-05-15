# Revisiting Black-box Ownership Verification for Graph Neural Networks

***Paper title: Revisiting Black-box Ownership Verification for Graph Neural Networks***

This paper will appear in IEEE Symposium on Security and Privacy 2024. 

This repo contains code that allows you to reproduce experiments presented in the paper.

## Environment Setup

Opearting system: Ubuntu 22.04.4 LTS

CPU: Intel i9-12900K

Graphics card: RTX 4090

RAM: 64GB

CUDA version: 11.8

You need to install some third-party libraries with the following commands:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install numpy
pip install scikit-learn
pip install tqdm
pip install pyyaml
pip install argparse
```

## File Illustration

### In "config" Folder:

1. **global_cfg.yaml: the config of overall experiment setting.** 
- target_model: architecture of target model. Valid values: [gcn, gat, sage].
- target_hidden_dims: hidden layer dimension of target model. Valid values: [[352, 128],[288, 128],[224, 128]].
- dataset: graph dataset. Valid values: [Cora, Citeseer, Amazon, DBLP, PubMed].
- train_setting: setting file for training local models. Valid values: fixed.
- test_setting: setting file for training real models. Valid values: [1, 2, 3, 4].
- embedding_dim: additional last layer in local and real models, used for reproducing Grove white-box method. Valid values: fixed.
- train_process: train local models. Valid values: fixed.
- test_process: train real models. Valid values: fixed.
- n_run: repeating number of verification experiments. Valid values: can be any number.
- train_save_root: saving path for the training models in ownership verification.
- test_save_root: saving path for the testing models in ownership verification.
- res_path: saving path for the ownership verification results.

2. **train_setting.yaml: the config of target model.**
- model_arches: architecture of model. Valid values: [gcn, gat, sage].
- layer_dims: hidden dimension of each layer can be selected. Valid values: [96,  160, 224, 288, 352]
- num_hidden_layers: number of hidden layer. Valid values: [2].
- num_model_per_arch: number of models will be trained for each model architecture. Valid values: can be any number.

3. **test_setting1.yaml/ test_setting2.yaml/ test_setting3.yaml/ test_setting4.yaml: the config of local models and test models.**
- model_arches: architecture of model. Valid values: [gcn, gat, sage] for setting 1/2, [gin, sgc] for setting 3/4.
- layer_dims: hidden dimension of each layer can be selected. Valid values: [96,  160, 224, 288, 352] for setting 1/3, [128, 192, 256, 320, 384] for setting 2/4.
- num_hidden_layers: number of hidden layer. Valid values: [2] for setting 1/3, [1, 3] for setting 2/4.
- num_model_per_arch: number of models will be trained for each model architecture. Valid values: can be any number.

### In "main" Folder:

1. **benign.py: train independent models**
2. **boundary.py: mask training data**
3. **extraction.py: train extraction models**
4. **main.py: main entry to run all experiments**
5. **robustness.py: robustness techniques**
6. **verification_cfg.py: run ownership verification experiment**

### In "model" Folder:

1. **extraction_models.py: architecture of extraction models**
2. **gnn_models.py: architecture of independent models**
3. **mlp.py: architecture of ownership verification classifier**

### In "utils" Folder:

1. **config.py:**
- data_path: saving path for graph dataset.
- task_type: learning paradigms. Valid values: [transductive, inductive]
- split_dataset_ratio: ratio of splitting training data and testing data for each kind of models.
- mask_feat_ratio: ratio of all nodes in the training data will be masked.
- prune_weight_ratio: ratio of model parameters will be pruned in robustness experiments.
- benign_train_epochs: number of epochs to train independent models.
- benign_lr: learning rate of training independent models.
- extraction_train_epochs: number of epochs to train extraction models.
- extraction_lr: learning rate of training extraction models.
- extraction_method: access to model extraction. Valid values: [black_box, white_box]
2. **datareader.py: load graph dataset and construct self-defined data**
3. **graph_operator.py: split dataset for inductive learning**


## Run Experiments

### Base Ownership Verification Experiments (See Section 5.2)

Everytime after setting the corresponding parameters in configurations, you can directly run the "main.py" file under the "main" folder.

```
python main.py
```

For example, with the default settings, you will run four verification settings (see Section 4.2), with no masking magnitude, in the Cora dataset, under transductive learning.

### Extended Studies (See Section 6.3)

1. extended study I: impact of independent models which are trained on randomly picked data

Inside "train_models_by_setting" function in "verification_cfg.py": set "process" parameter to "test".

```
python main.py
```

2. extended study II: impact of local model numbers

Inside "train_setting1.yaml": set the "num_model_per_arch" parameter to the local model numbers you want to test. In our paper, valid values: [10, 20, 30, 40, 50].

```
python main.py
```

### Robustness Experiments (See Section 6.4)

Fine-tune, prune and double extraction are all implemented in the "robustness.py" file. 

To run all three robustness techniques, you need to pass the saving path of real test models to the functions.

Besides, for the prune robustness experiment, you need to set the magnitude of prune.

After setting corresponding parameters in configurations, you can directly run the "robustness.py" file under the "main" folder to get real test models after robustness techniques.

```
python robustness.py
```

And then you should change the "test_save_root" parameter to the path where you saved the real test models after robustness techniques and run the ownership verification experiment again.

```
python main.py
```

### Adaptive Attacks (See Section 6.4)

Inside "train_models_by_setting" function in "verification_cfg.py": set "classifier" parameter to "classifier_model".

```
python main.py
```

## Results Viewing

All models and results will be saved in the path you set in the "global_cfg.yaml" file.

The name of each json file is the target model architecture.

Inside the file:

- TN: true negative number.
- TP: true positive number.
- FN: false negative number.
- FP: false positive number.
- FPR: false positive rate.
- FNR: false negative rate.
- Accuracy: accuracy of ownership verification.
- original_model_acc: target model accuracy of downstream task.
- mask_model_acc: masked target model accuracy of downstream task.
- train_inde_acc_list: local independent models accuracy of downstream task.
- train_surr_acc_list: local extraction models accuracy of downstream task.
- train_surr_fidelity_list: local extraction models fidelity of downstream task.
- test_inde_acc_list: real test independent models accuracy of downstream task.
- test_surr_acc_list: real test extraction models accuracy of downstream task.
- test_surr_fidelity_list: real test extraction models fidelity of downstream task.
- total_time: total running time.
- mask_run_time: masking running time.
## Citation
If you find several components of this work useful or want to use this code in your research, please cite the following paper:
@inproceedings{zhou2024revisiting,\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={Revisiting Black-box Ownership Verification for Graph Neural Networks},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Zhou, Ruikai and Yang, Kang and Wang, Xiuling and Wang, Wendy Hui and Xu, Jun},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;booktitle={2024 IEEE Symposium on Security and Privacy (SP)},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pages={210--210},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2024},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;organization={IEEE Computer Society}\
}
