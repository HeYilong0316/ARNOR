import tensorflow as tf
from bert import modeling

import pandas as pd
import argparse
from xsql2 import xsql, logger
from util import data_loader
import os
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


parser = argparse.ArgumentParser(prog='myprogram', description='Foo')
parser.add_argument('--bert_config_path', default='../publish/bert_config.json', type=str)
parser.add_argument('--bert_ckpt_path', default='../publish/bert_model.ckpt', type=str)
parser.add_argument('--val_db_path', default='../datas/val.db', type=str)
parser.add_argument('--lr_method', default='adam', type=str)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--clip', default=5, type=float)
parser.add_argument('--restore', default=True, type=bool)
parser.add_argument('--use_bias', default=False, type=bool)
parser.add_argument('--col_loss', default=True, type=bool)
args=parser.parse_args()
print(args)





val_data =data_loader('../datas/val.json', '../datas/val.tables.json', '../publish/vocab.txt', 217, 'val')



val_data.load()

val_graph = tf.Graph()

with val_graph.as_default() as g:
    val_model = xsql(args, 'val')
    val_model.build(g)



cur_score, accs, acc = val_model.run_evaule(val_data)



logger.info('acc: {}'.format(accs))
logger.info('aaa: {}'.format(acc))
for k, v in cur_score.items():
    logger.info('{}: P: {}, R: {}, F1: {}'.format(k, v['P'], v['R'], v['F1']))

logger.info('************\n')

