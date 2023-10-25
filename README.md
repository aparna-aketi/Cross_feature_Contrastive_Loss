# Cross-feature Contrastive Loss
This code is related to the paper titled, "Cross-feature Contrastive Loss for Decentralized Deep Learning on Heterogeneous Data". [Available here.](https://arxiv.org/abs/2310.15890) 

### Abstract
The current state-of-the-art decentralized learning algorithms mostly assume the data distribution to be Independent and Identically Distributed (IID). However, in practical scenarios, the distributed datasets can have significantly heterogeneous data distributions across the agents. In this work, we present a novel approach for decentralized learning on heterogeneous data, where knowledge distillation through contrastive loss on cross-features is utilized to improve performance. Cross-features for a pair of neighboring agents are the features (i.e., last hidden layer activations) obtained from the data of an agent with respect to the model parameters of the other agent. We demonstrate the effectiveness of the proposed technique through an exhaustive set of experiments on various Computer Vision datasets (CIFAR-10, CIFAR-100, Fashion MNIST, ImageNette, and ImageNet), model architectures, and network topologies. Our experiments show that the proposed method achieves superior performance (0.2-4%improvement in test accuracy) compared to other existing techniques for decentralized learning on heterogeneous data.

# Available Models
* ResNet
* VGG-11
* MobileNet-V2
* LeNet-5

# Available Datasets
* CIFAR-10
* CIFAR-100
* Fashion MNIST
* Imagenette
* ImageNet

# Available Graph Topologies
* Ring Graph
* Petersen Graph
* Dyck Graph
* Torus Graph
* Fully Connected Graph

# Requirements
* found in environment.yml file

# Hyper-parameters
* --world_size  = total number of agents
* --graph       = graph topology (default ring); options: [ring, dyck, petersen, torus, full]
* --neighbors   = number of neighbor per agent (default 2)
* --arch        = model to train
* --normtype    = type of normalization layer
* --dataset     = dataset to train; ; options: [cifar10, cifar100, fmnist, imagenette]
* --batch_size  = batch size for training (batch_size = batch_size per agent x world_size)
* --epochs      = total number of training epochs
* --lr          = learning rate
* --momentum    = momentum coefficient
* --gamma       = averaging rate for gossip 
* --skew        = amount of skew in the data distribution (alpha of Dirichlet distribution); 0.01 = completely non-iid and 10 = more towards iid
* --lambda_m    = model-variant contrastive loss coefficient
* --lambda_d    = data-variant contrastive loss coefficient

# How to run?

test file contains the commands to run the proposed algorithm and baselines on various datasets, models and graphs
```
sh test.sh
```

Some sample commands:

ResNet-20 with 16 agents ring topology with CCL:
```
python trainer.py --lr=0.1  --batch-size=512  --world_size=16 --skew=0.01 --gamma=1.0 --normtype=evonorm --lambda_m=0.1  --lambda_d=0.1 --epochs=200 --arch=resnet --momentum=0.9 --qgm=1 --weight_decay=1e-4  --nesterov --graph=ring --neighbors=2 --depth=20 --dataset=cifar10 --classes=10 --devices=4 --seed=123

```

## References
If you use the code, please cite the following paper:

```
@inproceedings{aketi2024ccl,
  title     = {Cross-feature Contrastive Loss for Decentralized Deep Learning on Heterogeneous Data},
  author    = {Aketi, Sai Aparna and Roy, Kaushik},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  url       = {https://arxiv.org/abs/2310.15890},
  year      = {2024}
}
```