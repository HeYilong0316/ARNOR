import tensorflow as tf
from bert import modeling

import pandas as pd
import argparse
from xsql2 import xsql, logger
from util import data_loader
import os
import json
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


parser = argparse.ArgumentParser(prog='myprogram', description='Foo')
parser.add_argument('--bert_config_path', default='../publish/bert_config.json', type=str)
parser.add_argument('--bert_ckpt_path', default='../publish/bert_model.ckpt', type=str)
parser.add_argument('--val_db_path', default='../datas/val.db', type=str)
parser.add_argument('--lr_method', default='adam', type=str)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batchsize', default=16, type=int)
parser.add_argument('--clip', default=5, type=float)
parser.add_argument('--restore', default=True, type=bool)
parser.add_argument('--use_bias', default=False, type=bool)
parser.add_argument('--col_loss', default=True, type=bool)
args=parser.parse_args()
print(args)


test_data = data_loader('../datas/val.json', '../datas/val.tables.json', '../publish/vocab.txt', 200, 'test')
#test_data =data_loader('./datas/test.json', './datas/test.tables.json', tokenizer, 342, 'test')


test_data.load(use_small=100)
test_graph = tf.Graph()


with test_graph.as_default() as g:
    test_model = xsql(args, 'test')
    test_model.build(g)

test_model.run_predict(test_data)
out = ''




# for pre in pres:
#     out+=json.dumps(pre, ensure_ascii=False, cls=NpEncoder)+'\n'
#
# with open('./result_test.json', 'w', encoding='utf-8') as w:
#     w.write(out)