# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" ComiRec
Reference:
    "Controllable Multi-Interest Framework for Recommendation"
    Cen et al., KDD'2020.
CMD example:
    python main.py --model_name ComiRec --emb_size 64 --lr 1e-3 --l2 1e-6 --attn_size 8 --K 4 --add_pos 1 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from models.BaseModel import SequentialModel
from utils import layers


class ComiRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden intent.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.add_pos = args.add_pos
        self.max_his = args.history_max
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.add_pos:
            self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.W1 = nn.Linear(self.emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        if self.add_pos:
            position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
            pos_vectors = self.p_embeddings(position)
            his_pos_vectors = his_vectors + pos_vectors
        else:
            his_pos_vectors = his_vectors

        # Self-attention
        attn_score = self.W2(self.W1(his_pos_vectors).tanh())  # bsz, his_max, K
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)
        attn_score = attn_score.transpose(-1, -2)  # bsz, K, his_max
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # bsz, K, emb

        i_vectors = self.i_embeddings(i_ids)
        if feed_dict['phase'] == 'train':
            target_vector = i_vectors[:, 0]  # bsz, emb
            target_pred = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K
            idx_select = target_pred.max(-1)[1]  # bsz
            user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        else:
            prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)  # bsz, -1, K
            prediction = prediction.max(-1)[0]  # bsz, -1

        return {'prediction': prediction.view(batch_size, -1)}


class ComiRecCPR(ComiRec):
    reader = 'CPRSeqReader'
    runner = 'CPRRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K', 'dyn_sample_rate', 'choose_rate', 'k_samples']


    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--dyn_sample_rate', type=float, default=1.5,
                            help='Dynamic sampling rate.')
        parser.add_argument('--choose_rate', type=float, default=2,
                            help='Choosing rate.')
        parser.add_argument('--k_samples', type=int, default=2,
                            help='k samples for CPR.') 
            
        return ComiRec.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.dyn_sample_rate = args.dyn_sample_rate
        self.choose_rate = args.choose_rate
        self.k_samples = args.k_samples

    
    class Dataset(ComiRec.Dataset):
        def actions_after_train(self):
            '''不生成负样本'''
            pass
        
        def _get_feed_dict(self, index):
            '''
                返回第 index 个样本的 feed_dict
                train时没有 neg_items，test/dev时有
            '''
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]

            if self.phase == 'train':
                item_ids = target_item
            else:
                if self.model.test_all:
                    neg_items = np.arange(1, self.corpus.n_items)
                else:
                    neg_items = self.data['neg_items'][index]
                item_ids = np.concatenate([[target_item], neg_items]).astype(int)

            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }

            pos = self.data['position'][index]
            user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
            if self.model.history_max > 0:
                user_seq = user_seq[-self.model.history_max:]
            feed_dict['history_items'] = np.array([x[0] for x in user_seq])
            feed_dict['history_times'] = np.array([x[1] for x in user_seq])
            feed_dict['lengths'] = len(feed_dict['history_items'])

            return feed_dict
        
        def collate_batch_base(self, feed_dicts):
            '''原版 collate_batch'''
            return super().collate_batch(feed_dicts)


        def collate_batch(self, feed_dicts):
            '''
                返回一个 batch 的 feed_dict，只在 train 时的DataLoader中调用
            '''
            feed_dicts = super().collate_batch(feed_dicts)
            # feed_dicts: [{"user_id": Tensor(user_id) (batchsize), 
            # 				"item_id": Tensor(item_id) (batchsize)},
            #               "history_items": Tensor(history_items) (batchsize, history_max), 
            #               'lengths': Tensor(lengths) (batchsize),
            # 				"batch_size": batch_size, "phase": phase,]

            batch_size = int(feed_dicts["batch_size"])
            sample_num = int(batch_size * self.model.dyn_sample_rate * self.model.choose_rate)

            # 采样 [sample_num, k_samples] Random Tensor
            samples = torch.stack([torch.randperm(batch_size)[:self.model.k_samples] for _ in range(sample_num)])       
            users_sample_list = list()
            pos_items_sample_list = list()
            history_sample_list = list()
            length_sample_list = list()

            for sample in samples:
                users_sample = feed_dicts["user_id"][sample]			# [k_samples] Tensor
                pos_items_sample = feed_dicts["item_id"][sample]		# [k_samples] Tensor
                history_sample = feed_dicts["history_items"][sample]	# [k_samples, history_max] Tensor
                length_sample = feed_dicts["lengths"][sample]			# [k_samples] Tensor
                flag = False

                # 检查是否错位负样本
                for i in range(self.model.k_samples):
                    if pos_items_sample[(i+1)%self.model.k_samples].item() in self.corpus.train_clicked_set[users_sample[i].item()]:
                        flag = True
                        break
                if flag:
                    continue
                
                users_sample_list.append(users_sample)
                pos_items_sample_list.append(pos_items_sample)
                history_sample_list.append(history_sample)
                length_sample_list.append(length_sample)

                if len(users_sample_list) == int( batch_size * self.model.dyn_sample_rate ):
                    break

            users_sample = torch.stack(users_sample_list)
            pos_items_sample = torch.stack(pos_items_sample_list)
            history_sample = torch.stack(history_sample_list)
            length_sample = torch.stack(length_sample_list)

            feed_dicts["user_id"] = users_sample            # [batch_size * dyn_sample_rate, k_samples]
            feed_dicts["item_id"] = pos_items_sample        # [batch_size * dyn_sample_rate, k_samples]
            feed_dicts["history_items"] = history_sample    # [batch_size * dyn_sample_rate, k_samples, history_max]
            feed_dicts["lengths"] = length_sample           # [batch_size * dyn_sample_rate, k_samples]

            return feed_dicts

    def inference(self, feed_dict):
        out_dict = ComiRec.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}

    def forward(self, feed_dict):
        pos_pred = list()
        neg_pred = list()
        for i in range(self.k_samples):
            feed_dict_i = copy.deepcopy(feed_dict)
            feed_dict_i['history_items'] = feed_dict_i['history_items'][:, i, :]
            feed_dict_i['lengths'] = feed_dict_i['lengths'][:, i]
            if i != self.k_samples - 1:
                feed_dict_i['item_id'] = feed_dict_i['item_id'][:, i:i+2]
            else:
                feed_dict_i['item_id'] = torch.cat([feed_dict_i['item_id'][:, i:], feed_dict_i['item_id'][:, :1]], dim=-1)
        
            # feed_dict_i: {"history_items": Tensor(history_items)  (batch_size * dyn_sample_rate, history_max),
            # 				"item_id": Tensor(item_id)              (batch_size * dyn_sample_rate, 2),
            # 				"lengths": Tensor(lengths)              (batch_size * dyn_sample_rate),}

            prediction = ComiRec.forward(self, feed_dict_i)['prediction']   # [batch_size * dyn_sample_rate, 2]
            pos_pred.append(prediction[:, 0])   # [batch_size * dyn_sample_rate]
            neg_pred.append(prediction[:, 1])   # [batch_size * dyn_sample_rate]

        pos_pred = torch.stack(pos_pred, dim=-1)  # [batch_size * dyn_sample_rate, k_samples]
        neg_pred = torch.stack(neg_pred, dim=-1)  # [batch_size * dyn_sample_rate, k_samples]

        return {'pos_pred': pos_pred, 'neg_pred': neg_pred, 'batch_size': feed_dict['batch_size']}


    def loss(self, out_dict):
        
        batch_size = out_dict['batch_size']
        pos_pred, neg_pred = out_dict['pos_pred'], out_dict['neg_pred']
        
        CPR_obj = pos_pred.mean(dim=-1) - neg_pred.mean(dim=-1)	# [batch_size * dyn_sample_rate]
        
        if CPR_obj.shape[0] > batch_size:
            top_batch_size_CPR, _ = torch.topk(-CPR_obj, batch_size)
        else:
            top_batch_size_CPR = -CPR_obj
        # 计算 CPR Loss
        CPRLoss = torch.nn.Softplus()(top_batch_size_CPR).mean()
        
        return CPRLoss



