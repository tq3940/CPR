# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" NeuMF
Reference:
    "Neural Collaborative Filtering"
    Xiangnan He et al., WWW'2017.
Reference code:
    The authors' tensorflow implementation https://github.com/hexiangnan/neural_collaborative_filtering
CMD example:
    python main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 \
    --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn

from models.BaseModel import GeneralModel, CPRModel


class NeuMF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'layers']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.layers = eval(args.layers)
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mlp_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mlp_i_embeddings = nn.Embedding(self.item_num, self.emb_size)

        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.emb_size
        for i, layer_size in enumerate(self.layers):
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.prediction = nn.Linear(pre_size + self.emb_size, 1, bias=False)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}


class NeuMFCPR(NeuMF, CPRModel):
    reader = 'CPRReader'
    runner = 'CPRRunner'
    extra_log_args = ['emb_size', 'layers', 'dyn_sample_rate', 'choose_rate', 'k_samples']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of each layer.")
        return CPRModel.parse_model_args(parser)
 
    def __init__(self, args, corpus):
        CPRModel.__init__(self, args, corpus)
        self.emb_size = args.emb_size
        self.layers = eval(args.layers)
        self._define_params()
        self.apply(self.init_weights)

    def inference(self, feed_dict):
        out_dict = NeuMF.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}

    def pre(self, u_ids, i_ids):
        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mlp_u_vectors = self.mlp_u_embeddings(u_ids)
        mlp_i_vectors = self.mlp_i_embeddings(i_ids)

        mf_vector = mf_u_vectors * mf_i_vectors
        mlp_vector = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)
        for layer in self.mlp:
            mlp_vector = layer(mlp_vector).relu()
            mlp_vector = self.dropout_layer(mlp_vector)

        output_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.prediction(output_vector)
        return prediction.view(u_ids.shape[0], -1)
    
    def forward(self, feed_dict):
        user, pos_items = feed_dict['user_id'], feed_dict['item_id']
        # user:  [batch_size * dyn_sample_rate, k_samples] Tensor
        # items: [batch_size * dyn_sample_rate, k_samples] Tensor
        pos_pred = list()
        neg_pred = list()
        for i in range(self.k_samples):
            feed_dict_i = dict()
            feed_dict_i['user_id'] = user[:, i]   # [batch_size * dyn_sample_rate]
            feed_dict_i['batch_size'] = user.shape[0]
            if i != self.k_samples - 1:
                feed_dict_i['item_id'] = pos_items[:, i:i+2]
            else:
                feed_dict_i['item_id'] = torch.cat([pos_items[:, i:], pos_items[:, :1]], dim=-1)
        
            prediction = NeuMF.forward(self, feed_dict_i)['prediction']   # [batch_size * dyn_sample_rate, 2]
            pos_pred.append(prediction[:, 0])   # [batch_size * dyn_sample_rate]
            neg_pred.append(prediction[:, 1])   # [batch_size * dyn_sample_rate]

        pos_pred = torch.stack(pos_pred, dim=-1)  # [batch_size * dyn_sample_rate, k_samples]
        neg_pred = torch.stack(neg_pred, dim=-1)  # [batch_size * dyn_sample_rate, k_samples]
        # pos_pred = self.pre(user, pos_items)
        # neg_pred = self.pre(user, neg_items)

        return {"pos_pred": pos_pred, "neg_pred": neg_pred, "batch_size": feed_dict["batch_size"]}

    # def loss(self, output_dict):
    #     loss1 = CPRModel.loss(self, output_dict)
    #     output_dict['prediction'] = torch.cat([output_dict['pos_pred'], output_dict['neg_pred']], dim=-1)
    #     loss2 = NeuMF.loss(self, output_dict)
    #     loss = 0.2 * loss1 + 0.8 * loss2
    #     return loss