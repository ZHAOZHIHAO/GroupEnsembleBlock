from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupEnsembleBlock(nn.Module):
    def __init__(self, input_length=2048, output_length=512,
                 group_num=32, subspace_total=1024, parallel_trans='linear'):
        super(GroupEnsembleBlock, self).__init__()
        self.group_num = group_num
        self.subspace_total = subspace_total
        self.mask_len = input_length // group_num
        if parallel_trans == 'linear':
            self.parallel_trans = nn.Linear(self.mask_len, output_length // group_num)
        elif parallel_trans == 'MLP':
            inter = output_length // group_num
            self.parallel_trans = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(self.mask_len, inter)),
                ('relu1', nn.ReLU()),
                ('linear2', nn.Linear(inter, inter)),
                ('relu2', nn.ReLU()),
                ('linear3', nn.Linear(inter, output_length // group_num),)]))
        else:
            print('unsupported parallel transformation')

        self.bn_after_cat = nn.BatchNorm1d(output_length)
        self.mask_pos = torch.nn.Parameter(data=torch.randint(high=input_length, size=(self.subspace_total * self.mask_len,)),
                                           requires_grad=False)
        self.float_mask = torch.nn.Parameter(data=torch.Tensor(1, group_num, self.subspace_total // group_num,
                                             self.mask_len), requires_grad=False)
        self.float_mask.data.uniform_(0, 1)
        self.tau = 0.1

        print("Parameters in group ensemble block:")
        print("self.mean_num ", group_num)
        print("self.mask_num ", subspace_total)
        print("self.mask_len", self.mask_len)
        print("Sampled elements amount,", len(list(set(self.mask_pos.tolist()))))
        print("self.mask_pos.is_leaf", self.float_mask.is_leaf)
        print("Using float mask")
        print("tau for masking b_n", self.tau)

    def forward(self, x):
        x = torch.index_select(x, dim=-1, index=self.mask_pos)
        x = x.view(x.shape[0], self.group_num, self.subspace_total // self.group_num, self.mask_len)
        x = x * F.relu(self.float_mask - self.tau) / (1.0 - self.tau)
        x = self.parallel_trans(x)
        x = torch.mean(x, dim=2)
        x = x.view(x.shape[0], -1)
        x = self.bn_after_cat(x)
        return x
