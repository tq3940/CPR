# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner


class CPRRunner(BaseRunner):
	def fit(self, dataset, epoch=-1) -> float:
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)

		model.train()
		loss_lst = list()

		try:
			dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
							collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		except KeyboardInterrupt:
			logging.info("数据加载被中断")
			# 确保所有工作线程正确终止
			if hasattr(dl, 'worker_pids_set'):
				for w in dl.workers:
					w.terminate()
			raise

		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)

			# 去除了训练前打乱项目
			model.optimizer.zero_grad()
			out_dict = model(batch)

			loss = model.loss(out_dict)
			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())

		return np.mean(loss_lst).item()
	
	def predict(self, dataset, save_prediction: bool = False) -> np.ndarray:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.model.eval()
		predictions = list()

		# 只修改了collate_fn为原版的collate_batch_base
		# dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
		# 				collate_fn=dataset.collate_batch_base, pin_memory=self.pin_memory)
		try:
			dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
							collate_fn=dataset.collate_batch_base, pin_memory=self.pin_memory)
		except KeyboardInterrupt:
			logging.info("数据加载被中断")
			# 确保所有工作线程正确终止
			if hasattr(dl, 'worker_pids_set'):
				for w in dl.workers:
					w.terminate()
			raise

		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			if hasattr(dataset.model,'inference'):
				prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			else:
				prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			predictions.extend(prediction.cpu().data.numpy())
		predictions = np.array(predictions)

		if dataset.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf
		return predictions
