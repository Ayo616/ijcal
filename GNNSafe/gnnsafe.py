import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from backbone import *
import numpy as np
from logger import save_cust_result, save_cust_result2
from layers import GraphAttentionLayer,EnergyGAT
from torch.utils.data import RandomSampler

class GNNSafe(nn.Module):
    '''
    The model class of energy-based models for out-of-distribution detection
    The parameter args.use_reg and args.use_prop control the model versions:
        Energy: args.use_reg = False, args.use_prop = False
        Energy FT: args.use_reg = True, args.use_prop = False
        GNNSafe: args.use_reg = False, args.use_prop = True
        GNNSafe++ args.use_reg = True, args.use_prop = True
    '''
    def __init__(self, d, c, args):
        super(GNNSafe, self).__init__()
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
        '''return predicted logits'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index,recession, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
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
            # yan
            if recession == 1:
                #########################################衰减#####################################################
                avg_e = torch.mean(e)
                # 设置当e小于平均e时，将其缩减20%
                mask = e < avg_e
                e[mask] = e[mask] * 0.8


                # 获取周围节点的e值
                neighbor_e = e[row]
                # 计算周围节点的e平均值
                avg_neighbor_e = torch.mean(neighbor_e, dim=1)
                # 设置当e小于周围节点的e平均值时，将其缩减20%
                # import pdb
                # pdb.set_trace()
                mask = e < avg_neighbor_e.mean()
                e[mask] = e[mask] * 0.8

                # 随机挑选N个节点
                import random
                num_nodes = e.shape[0]
                indices = random.sample(range(num_nodes), 10)
                # import pdb
                # pdb.set_trace()
                # 计算N个节点的平均值e
                neighbor_e_random = torch.mean(e[indices])

                # 设置低于N个节点的平均值e，则衰减20%
                mask = e >  2 * neighbor_e_random
                e[mask] = e[mask] + 2


                ###################################保持#############################################
                # 设置节点度大于平均节点度时，不更新e
                avg_d = torch.mean(d)
                mask = d > avg_d
                e[mask] = e[mask].detach()



        return e.squeeze(1)


    def detect(self, dataset, node_idx, device, args):
        '''return negative energy, a vector for all input nodes'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else: # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        if args.use_prop: # use energy belief propagation
            neg_energy = self.propagation(neg_energy, edge_index, args.recession,args.K, args.alpha)
            # neg_energy = self.propagation(neg_energy, edge_index, args.recession, args.K, args.alpha)
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        # get predicted logits from gnn classifier
        # 把所有的样本内数据输入到模型中得到结果
        logits_in = self.encoder(x_in, edge_index_in)
        # 把所有的OOD数据输入到模型中得到结果
        logits_out = self.encoder(x_out, edge_index_out)

        # 样本内训练集 和 OOD数据集
        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx

        # import pdb
        # pdb.set_trace()
        ######## compute supervised training loss
        # 常规的有监督学习计算误差
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
            if args.oodloss == 1:
                # yan 增加损失函数
                sup_loss += np.log(criterion(logits_out[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float)))

        else:
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

            if args.oodloss == 1:
            # yan 增加损失函数
            #     breakpoint()
                new_logits_out = logits_out[[i for i in RandomSampler(logits_out, num_samples=dataset_ind.y[train_in_idx].shape[0])]]
                pred_in = F.log_softmax(new_logits_out, dim=1)
                sup_loss += torch.abs(torch.log(criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))))

                # pred_in = F.log_softmax(logits_out[train_ood_idx], dim=1)
                # if True:
                # if torch.log(criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))) > 0:
                # #     dataset_ind.y[train_in_idx].squeeze(1)
                #     sup_loss += torch.abs(torch.log(criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))))
                #
                #     if logits_out[train_ood_idx].shape[0] > logits_in[train_in_idx].shape[0]:
                #         X = logits_out[train_ood_idx].split([logits_in[train_in_idx].shape[0],logits_out[train_ood_idx].shape[0]-logits_in[train_in_idx].shape[0]],dim=0)[0]
                #         Y = logits_in[train_in_idx]
                #         idx = [i for i in RandomSampler(X, num_samples=500)]
                #         idy = [i for i in RandomSampler(Y, num_samples=500)]
                #         X = X[idx]
                #         Y = Y[idy]
                #         mmd_loss = torch.abs(self.mmd_rbf(X,X) - self.mmd_rbf(X,Y))
                #
                #     sup_loss += mmd_loss
                    # print(torch.log(criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))))

        # yan prediction label sum
        # y_pred = pred_in.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
        # y_true = dataset_ind.y[train_in_idx].detach().cpu().numpy()
        # for i in range(dataset_ind.y[train_in_idx].shape[1]):
        #     is_labeled = y_true[:, i] == y_true[:, i]
        #     correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        #     print(np.sum(correct))
        #     from GNNSafe.logger import save_cust_result
        #     save_cust_result(np.sum(correct).astype(str),'mytest')

        # yan
        ind_result = self.detect(dataset_ind, dataset_ind.splits['train'], device, args).cpu()
        ood_result = self.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()
        result = np.max(ind_result.tolist()).astype(str)+','+ np.min(ind_result.tolist()).astype(str) +',' + np.max(ood_result.tolist()).astype(str) + ',' +np.min(ood_result.tolist()).astype(str)
        save_cust_result2(result, 'detect_In_Out_max_min')

        # 计算箱线图的五个关键值  无detect下能量的情况,ood
        min_val = np.min(ind_result.tolist()).astype(str)
        q1 = np.percentile(ind_result.tolist(), 25).astype(str)
        q2 = np.median(ind_result.tolist()).astype(str)
        q3 = np.percentile(ind_result.tolist(), 75).astype(str)
        max_val = np.max(ind_result.tolist()).astype(str)
        result = min_val + ',' + q1 + ',' + q2 + ',' + q3 + ',' + max_val
        save_cust_result2(result,'detect_ind_maxmin_energy_score')


        # 计算箱线图的五个关键值  无detect下能量的情况,ood
        min_val = np.min(ood_result.tolist()).astype(str)
        q1 = np.percentile(ood_result.tolist(), 25).astype(str)
        q2 = np.median(ood_result.tolist()).astype(str)
        q3 = np.percentile(ood_result.tolist(), 75).astype(str)
        max_val = np.max(ood_result.tolist()).astype(str)
        result = min_val + ',' + q1 + ',' + q2 + ',' + q3 + ',' + max_val
        save_cust_result2(result,'detect_ood_maxmin_energy_score')

        if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)

            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1) +5
        else: # for single-label multi-class classification
            # 把所有的样本内数据输入到模型中得到结果，转化为能量
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            # 把所有的OOD数据输入到模型中得到结果，转化为能量
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1) +5

            #yan 无detect下能量的情况
            # result = np.max(energy_in.tolist()).astype(str)+','+ np.min(energy_in.tolist()).astype(str) +',' + np.max(energy_out.tolist()).astype(str) + ',' +np.min(energy_out.tolist()).astype(str)
            # save_cust_result2(result, 'In_Out_max_min')
            #
            # # 计算箱线图的五个关键值  无detect下能量的情况, ind
            # min_val = np.min(energy_in.tolist()).astype(str)
            # q1 = np.percentile(energy_in.tolist(), 25).astype(str)
            # q2 = np.median(energy_in.tolist()).astype(str)
            # q3 = np.percentile(energy_in.tolist(), 75).astype(str)
            # max_val = np.max(energy_in.tolist()).astype(str)
            # result = min_val + ',' + q1 + ',' + q2 + ',' + q3 + ',' + max_val
            # save_cust_result2(result, 'In_max_min')
            #
            # # 计算箱线图的五个关键值  无detect下能量的情况,ood
            # min_val = np.min(energy_out.tolist()).astype(str)
            # q1 = np.percentile(energy_out.tolist(), 25).astype(str)
            # q2 = np.median(energy_out.tolist()).astype(str)
            # q3 = np.percentile(energy_out.tolist(), 75).astype(str)
            # max_val = np.max(energy_out.tolist()).astype(str)
            # result = min_val + ',' + q1 + ',' + q2 + ',' + q3 + ',' + max_val
            # save_cust_result2(result, 'Out_max_min')

            # 无detect下每个节点能量的情况
            # indexs = [5,80,300,400,600,800]
            # result = ''
            # for i in indexs:
            #     if result == '':
            #         result += str(energy_in.tolist()[i])+ ',' +str(energy_out.tolist()[i])
            #     else:
            #         result += ','+ str(energy_in.tolist()[i]) + ',' + str(energy_out.tolist()[i])
            # save_cust_result2(result,'randomNodeEnerge')

            loss = sup_loss

        return loss

    def guassian_kernel(self,source, target, kernel_mul=4.0, kernel_num=4, fix_sigma=None):
        '''
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params:
    	    source: 源域数据（n * len(x))
    	    target: 目标域数据（m * len(y))
    	    kernel_mul:
    	    kernel_num: 取不同高斯核的数量
    	    fix_sigma: 不同高斯核的sigma值
    	Return:
    		sum(kernel_val): 多个核矩阵之和
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
        # 将total复制（n+m）份
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # 调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        # 高斯核函数的数学表达式
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)  # /len(kernel_val)

    def mmd_rbf(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        计算源域数据和目标域数据的MMD距离
        Params:
    	    source: 源域数据（n * len(x))
    	    target: 目标域数据（m * len(y))
    	    kernel_mul:
    	    kernel_num: 取不同高斯核的数量
    	    fix_sigma: 不同高斯核的sigma值
    	Return:
    		loss: MMD loss
        '''
        batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        # 根据式（3）将核矩阵分成4部分
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

