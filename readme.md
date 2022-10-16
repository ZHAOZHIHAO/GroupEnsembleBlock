# Group ensemble block: subspace diversity improves coarse-to-fine image retrieval

> A pytorch implementation.

[**Group ensemble block: subspace diversity improves coarse-to-fine image retrieval**](https://ieeexplore.ieee.org/document/9889202)  
Zhihao Zhao, Shangqing Zhao, [Samuel Cheng](https://samuelcheng.info/)
IEEE Transactions on Artificial Intelligence, 2022. 

Please use this bibtex to cite this repository:
```tex
@article{zhao2022group,
  title={Group ensemble block: subspace diversity improves coarse-to-fine image retrieval},
  author={Zhao, Zhihao and Zhao, Shangqing and Cheng, Samuel},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2022},
  publisher={IEEE}
}
```

### Demo usage
As in this example, you just import the ''GroupEnsembleBlock'' class, and use it as a regular linear layer.
For example,
```python
from src.group_ensemble_block import GroupEnsembleBlock

class Net(nn.Module):
    def __init__(self):
        ...
        # Pay attention that 
        # a. the parameter group_num should be i) smaller than output length is) a factor of output length
        # b. the parameter subspace_total should be N * group_num, where N can be any interage larger than 1 because subspace_total = group_num * N
        # c. this parameter set is used for CIFAR-10 classification: 10 groups, 10 subspaces in each group, and 100 subspaces in total.
        self.ensemble_block = GroupEnsembleBlock(input_length=512, output_length=10, group_num=10, subspace_total=100)
    
    def forward(self,x):
        ...
        # use it as a regular linear layer, e.g., self.linear(x)
        x = self.ensemble_block(x)
        return x
```

### Usage for CIFAR-10 classification
Please see the Google Colab example, CIFAR10Classification_GroupEnsembleBlock.ipynb 

### Usage for image retrieval 
Any file other than the colab file is for image retrieval.

##### Datasets
Fill the empty folders under the *data* folder by the tree structure files in  the *data* folder.
Note: We show the dataset structure of CIFAR-100 and ImageNet-C16 for coarse-to-fine image retrieval, in the *data* folder. After downloading these open source datasets, you can rearrange them into the dataset structure as in the *data* folder. The datasets are not directly provided here as they are large.

##### Prerequisites
This code is tested with the following package verisions. A higher pytorch verision may potentially cause some problems for using the NCA.py and LinearAverage.py. But it's totally okay to use any pytorch version if you solely use the group ensemble block in group_ensemble_block.py.
- Python 3.7
- Pytorch 1.7.0
- Torchvision  0.8.0
- CUDA 10.2

##### Train and test ImageNet-C16
```bash
python main.py --TRAIN --dataset ImageNet-C16 --checkpoint_folder ./checkpoints
```
```bash
python main.py --TEST --dataset ImageNet-C16 --load_checkpoint_path ./checkpoints/epoch313.pth
```

##### 
Feel free to contact me for any problems in running the codes.