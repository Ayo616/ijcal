import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from backbone import *
import numpy as np
from logger import *
from torch.utils.data import RandomSampler

class GEVN(nn.Module):
    def __init__(self, d, c, args):
        super(GEVN, self).__init__()

        self.encoder_2_layer = GCN(in_channels=d,
                                   hidden_channels=args.hidden_channels,
                                   out_channels=c,
                                   num_layers=args.num_layers - 1,
                                   dropout=args.dropout1,
                                   use_bn=args.use_bn)

        self.encoder_3_layer = GCN(in_channels=d,
                                   hidden_channels=args.hidden_channels,
                                   out_channels=c,
                                   num_layers=args.num_layers + 1,
                                   dropout=args.dropout3,
                                   use_bn=args.use_bn)
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn)

        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                        out_channels=c, num_layers=args.num_layers,
                        dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout, use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError


    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def recession(self, e, edge_index,arg, prop_layers=1, alpha=0.5):
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

        for _ in range(prop_layers):
            # yan
            #########################################衰减#####################################################
            avg_e = torch.mean(e)
            # 设置当e小于平均e时，将其缩减
            mask = e < avg_e
            # import pdb
            # pdb.set_trace()
            e[mask] = e[mask] * arg.reduce

            # # 获取周围节点的e值
            # neighbor_e = e[row]
            # # 计算周围节点的e平均值
            # avg_neighbor_e = torch.mean(neighbor_e, dim=1)
            # # 设置当e小于周围节点的e平均值时，将其缩减20%
            # # import pdb
            # # pdb.set_trace()
            # mask = e < avg_neighbor_e.mean()
            # e[mask] = e[mask] * arg.reduce

            # 随机挑选N个节点
            # import random
            # num_nodes = e.shape[0]
            # indices = random.sample(range(num_nodes), 10)
            # # import pdb
            # # pdb.set_trace()
            # # 计算N个节点的平均值e
            # neighbor_e_random = torch.mean(e[indices])
            # # 设置大于N个节点的平均值e的2倍，则衰减20%
            # mask = e >  3 * neighbor_e_random
            # e[mask] = e[mask] + 3

            ###################################保持#############################################
            # 设置节点度大于平均节点度时，不更新e
            # avg_d = torch.mean(d)
            # mask = d > avg_d
            # e[mask] = e[mask].detach()
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.use_mlayer:
            logits += self.encoder_2_layer(x,edge_index)
            logits += self.encoder_3_layer(x, edge_index)

        if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else: # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)

        if args.use_sprop:
            neg_energy = self.propagation(neg_energy, edge_index,args.K, args.alpha)
        if args.use_srece:
            neg_energy = self.recession(neg_energy, edge_index, args,args.K, args.alpha)
        return neg_energy[node_idx]


    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        logits_in = self.encoder(x_in, edge_index_in)


        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)
        logits_out = self.encoder(x_out, edge_index_out)


        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        oodloss = 0
        ######## compute supervised training loss
        # 常规的有监督学习计算误差
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
            if args.oodloss:
                # yan 增加损失函数
                oodloss= np.log(criterion(logits_out[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float)))

        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

            if args.oodloss:
            # yan 增加损失函数
                new_logits_out = logits_out[[i for i in RandomSampler(logits_out, num_samples=dataset_ind.y[train_in_idx].shape[0])]]
                pred_in = F.log_softmax(new_logits_out, dim=1)
                oodloss = torch.abs(torch.log(criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))))


        # detect下能量的情况
        ind_result = self.detect(dataset_ind, dataset_ind.splits['train'], device, args).cpu()
        ood_result = self.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()

        save_boxplot_ood(ood_result,'detect_ood_maxmin_energy_score')
        save_boxplot_ind(ind_result, 'detect_ind_maxmin_energy_score')
        save_in_out_max_min(ind_result,ood_result,'detect_In_Out_max_min')


        if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1) +5

        # for single-label multi-class classification
        else:
            # 把所有的样本内数据输入到模型中得到结果，转化为能量
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            # 把所有的OOD数据输入到模型中得到结果，转化为能量
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1) +5

            # 无detect下能量的情况
            save_in_out_max_min(energy_in, energy_out, 'detect_In_Out_max_min')
            save_boxplot_ood(energy_out, 'detect_ood_maxmin_energy_score')
            save_boxplot_ind(energy_in, 'detect_ind_maxmin_energy_score')



        loss = sup_loss + args.lamda * oodloss

        return loss

 

