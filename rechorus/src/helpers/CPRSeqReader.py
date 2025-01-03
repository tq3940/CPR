# -*- coding: UTF-8 -*-

import logging
import pandas as pd

from helpers.SeqReader import SeqReader


class CPRSeqReader(SeqReader):
    def __init__(self, args):
        super().__init__(args)
        # 唯一修改：打乱数据集
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = self.data_df[key].sample(frac=1, replace=False).reset_index(drop=True)
