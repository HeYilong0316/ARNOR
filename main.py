# author: heyilong

from model.model import model
from model.data_loader import data_loader
from model.util import load_data_file, build_maps, clean, get_logger, check_env, parser_score

import os
import heapq
import json
import numpy as np
import tensorflow as tf

from collections import OrderedDict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


flags = tf.app.flags
flags.DEFINE_boolean("clean",                           False,          "Clean train file")
flags.DEFINE_boolean("train",                           False,          "Whether train the model")
flags.DEFINE_boolean("restore",                         False,          "Wither restore ckpt")
flags.DEFINE_boolean("use_small",                       False,          "Wither use small data")

# configurations for the model
flags.DEFINE_integer("type_dim",                        50,             "Embedding size for entity type")
flags.DEFINE_integer("position_dim",                    50,             "Embedding size for position")
flags.DEFINE_integer("word_dim",                        100,            "Embedding size for word")
flags.DEFINE_integer("lstm_dim",                        500,            "Num of hidden units in LSTM")
flags.DEFINE_integer("pos_max",                         100,            "Max position")

# configurations for training
flags.DEFINE_float("init_patterns_ratio",               0.1,            "Top percentage for init patterns")
flags.DEFINE_integer("init_patterns_max",               20,             "Max num of init patterns")
flags.DEFINE_float("pattern_threshold",                 0.5,            "Pattern threshold for update trustable pattern")
flags.DEFINE_integer("pattern_max",                     5,              'Max num of updating pattern')
flags.DEFINE_float("beta",                              1.,             "bate for attention regularization")
flags.DEFINE_integer("first_loop_epoch",                10,             "First loop epoch")
flags.DEFINE_integer("epoch",                           50,             "Epoch")
flags.DEFINE_float("clip",                              5.,            "Gradient clip")
flags.DEFINE_float("dropout",                           0.5,            "Dropout keep prob")
flags.DEFINE_integer("batchsize",                       160,            "Batch size")
flags.DEFINE_float("lr",                                0.001,          "Initial learning rate")
flags.DEFINE_string("optimizer",                        "adam",         "Optimizer for training")
flags.DEFINE_boolean("zero",                            True,           "Wither replace digits with zero")
flags.DEFINE_boolean("lower",                           False,          "Wither lower case")
flags.DEFINE_boolean("redistribution",                  True,           "Wither redistribution")
flags.DEFINE_boolean("attention_regularization",        True,           "Wither attention regularization")
flags.DEFINE_boolean("bootstrap",                       True,           "Wither bootstrap")
FLAGS = tf.app.flags.FLAGS





def get_config():
    config = OrderedDict()
    config['batchsize'] = FLAGS.batchsize
    config['word_dim'] = FLAGS.word_dim
    config['position_dim'] = FLAGS.position_dim
    config['type_dim'] = FLAGS.type_dim
    config['hidden_dim'] = FLAGS.lstm_dim
    config['pos_max'] = FLAGS.pos_max
    config['redistribution'] = FLAGS.redistribution
    config['attention_regularization'] = FLAGS.attention_regularization
    config['bootstrap'] = FLAGS.bootstrap
    config['zero'] = FLAGS.zero
    config['lower'] = FLAGS.lower
    config['first_loop_epoch'] = FLAGS.first_loop_epoch
    config['epoch'] = FLAGS.epoch
    config['init_patterns_ratio'] = FLAGS.init_patterns_ratio
    config['init_patterns_max'] = FLAGS.init_patterns_max
    config['pattern_threshold'] = FLAGS.pattern_threshold
    config['pattern_max'] = FLAGS.pattern_max
    config['beta'] = FLAGS.beta
    config['lr'] = FLAGS.lr
    config['clip'] = FLAGS.clip
    config['lr_method'] = FLAGS.optimizer
    config['restore'] = FLAGS.restore
    config['use_small'] = FLAGS.use_small
    config['dropout'] = FLAGS.dropout

    config['sel_label'] = ['/people/person/children',                   '/business/company/founders',
                           '/people/deceased_person/place_of_death',    '/people/person/place_of_birth',
                           '/location/neighborhood/neighborhood_of',    '/business/person/company',
                           '/people/person/place_lived',                '/location/country/capital',
                           '/people/person/nationality',                '/location/location/contains',
                           'None']

    # 原paper中做了这两个label的转换
    config['label_map'] = {'/location/country/administrative_divisions': '/location/location/contains'}


    return config


def train(config, logger):
    logger.info('------load train data------')
    train_datas = load_data_file('datas/train.json', zero=config['zero'], lower=config['lower'])
    config = build_maps(train_datas, config, logger)

    train_data_loader = data_loader(train_datas, config)
    data_count, badcase, ignore, pos_max = train_data_loader.load(config['use_small'])
    logger.info('---Data count---')
    for label, num in data_count.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Badcase count---')
    for label, num in badcase.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Ignore count---')
    for label, num in ignore.items():
        logger.info('{:<50}\t{:>}'.format(label, num))
    logger.info('--------------------------\n')

    logger.info('------load val data------')
    val_datas = load_data_file('datas/dev.json', zero=config['zero'], lower=config['lower'])
    val_data_loader = data_loader(val_datas, config)
    data_count, badcase, ignore, _ = val_data_loader.load(config['use_small'])
    logger.info('---Data count---')
    for label, num in data_count.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Badcase count---')
    for label, num in badcase.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Ignore count---')
    for label, num in ignore.items():
        logger.info('{:<50}\t{:>}'.format(label, num))
    logger.info('--------------------------\n')

    logger.info('------load test data------')
    test_datas = load_data_file('datas/test.json', zero=config['zero'], lower=config['lower'])
    test_data_loader = data_loader(test_datas, config)
    data_count, badcase, ignore, _ = test_data_loader.load(config['use_small'])
    logger.info('---Data count---')
    for label, num in data_count.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Badcase count---')
    for label, num in badcase.items():
        logger.info('{:<50}\t{:>}'.format(label, num))

    logger.info('---Ignore count---')
    for label, num in ignore.items():
        logger.info('{:<50}\t{:>}'.format(label, num))
    logger.info('--------------------------\n')

    config['position_num'] = pos_max
    print(config['position_num'], 'position_num')
    trustable_pattern = None
    if config['bootstrap'] or config['redistribution']:
        logger.info('init patterns')

        pattern_dict = {}
        for data in train_data_loader.dataset:
            pattern = data[-1]
            label = data[-2]
            if label not in pattern_dict:
                pattern_dict[label] = {}
            if pattern not in pattern_dict[label]:
                pattern_dict[label][pattern] = 0
            pattern_dict[label][pattern] += 1

        trustable_pattern = {}
        for k, v in pattern_dict.items():
            sel_num = int(config['init_patterns_ratio'] * len(v))
            sel_num = config['init_patterns_max'] if sel_num > config['init_patterns_max'] else sel_num
            v = heapq.nlargest(sel_num, v.items(), key=lambda x: x[1])
            trustable_pattern[k] = set([v1 for v1, v2 in v])

    with tf.Graph().as_default() as g:
        train_model = model(config, 'train', trustable_pattern)
        train_model.build(g)

    with tf.Graph().as_default() as g:
        val_model = model(config, 'val')
        val_model.build(g)


    # for not boostrap or firt loop of bootstrap
    best_score = [-1., 0, None, None]
    loop = config['first_loop_epoch'] if config['bootstrap'] else config['epoch']

    logger.info('not bootstrap or first loop of bootstrap:{}'.format(loop))
    for epoch in range(loop):

        logger.info('***TRAIN: {}***'.format(epoch))
        if config['attention_regularization']:
            attentions, kls, patterns, labels = train_model.run_train(train_data_loader)
        else:
            patterns, labels = train_model.run_train(train_data_loader)

        logger.info('***VAL***')
        cur_score, accs = val_model.run_evaule(val_data_loader)
        best_score, is_new = parser_score(epoch, best_score, cur_score, accs, logger)
        if is_new:
            logger.info('***TEST***')
            test_score, test_accs = val_model.run_evaule(test_data_loader)
            parser_score(epoch, '', test_score, test_accs, logger, 'test')
        logger.info('******\n')


    # for other bootstrap loop
    if config['bootstrap']:
        logger.info('for other bootstrap loop')

        # update patterns
        pattern_condidates = {}
        for kl, pattern, label in zip(kls, patterns, labels):
            pattern_score = 1. / (1. + kl)
            if pattern_score>config['pattern_threshold'] and pattern not in trustable_pattern:
                if label not in pattern_condidates:
                    pattern_condidates[label] = []
                pattern_condidates[label].append((pattern, pattern_score))

        print('for update patterns')
        for label in config['label_dict'].keys():
            if label not in pattern_condidates:
                continue
            num = len(pattern_condidates[label]) if len(pattern_condidates[label]) > config['pattern_max'] \
                else config['pattern_max']
            pattern_condidates_label = heapq.nlargest(num, pattern_condidates[label], key=lambda x: x[1])
            trustable_pattern[label].update(pattern_condidates_label)
        train_model.update_trustable_pattern(trustable_pattern)


        for epoch in range(config['epoch'] - config['first_loop_epoch']):
            logger.info('***TRAIN: {}***'.format(epoch))

            if config['attention_regularization']:
                attentions, kls, patterns, labels = train_model.run_train(train_data_loader)
            else:
                patterns, labels = train_model.run_train(train_data_loader)

            logger.info('***VAL***')
            cur_score, accs = val_model.run_evaule(val_data_loader)
            best_score, is_new = parser_score(epoch, best_score, cur_score, accs, logger)
            if is_new:
                logger.info('***TEST***')
                test_score, test_accs = val_model.run_evaule(test_data_loader)
                parser_score(epoch, '', test_score, test_accs, logger, 'test')
            logger.info('******\n')

        # update patterns
        pattern_condidates = {}
        for kl, pattern, label in zip(kls, patterns, labels):
            pattern_score = 1 / (1 + kl)
            if pattern_score>config['pattern_threshold'] and pattern not in trustable_pattern:
                if label not in pattern_condidates:
                    pattern_condidates[label] = []
                pattern_condidates[label].append((pattern, pattern_score))

        print('for update patterns')
        for label in config['label_dict'].keys():
            if label not in pattern_condidates:
                continue
            num = len(pattern_condidates[label]) if len(pattern_condidates[label]) > config['pattern_max'] \
                else config['pattern_max']
            pattern_condidates_label = heapq.nlargest(num, pattern_condidates[label], key=lambda x: x[1])
            trustable_pattern[label].update(pattern_condidates_label)
        train_model.update_trustable_pattern(trustable_pattern)


def test(config, logger):

    print('load test data')

    if os.path.exists('maps.npy'):
        vocab, type_dict, label_dict = np.load('maps.npy')
    else:
        raise FileNotFoundError('can not find maps.npy')

    for k, v in config.items():
        logger.info('config {}: {}'.format(k, v))

    config['vocab'] = vocab
    config['type_dict'] = type_dict
    config['label_dict'] = label_dict

    config['word_num'] = len(vocab)
    config['type_num'] = len(type_dict)
    config['label_num'] = len(label_dict)
    config['position_num'] = config['pos_max']

    test_datas = load_data_file('datas/test.json', zero=config['zero'], lower=config['lower'])
    test_data_loader = data_loader(test_datas, config)
    badcase = test_data_loader.load()
    logger.info('test datas: {}'.format(len(test_data_loader)))
    logger.info('test badcase: {}'.format(badcase))


    with tf.Graph().as_default() as g:
        test_model = model(config, 'test')
        test_model.build(g)


    logger.info('***TEST***')
    test_score, test_accs = test_model.run_evaule(test_data_loader)
    parser_score(0, '', test_score, test_accs, logger, 'test')
    logger.info('******\n')



def main(_):
    logger = get_logger()
    if FLAGS.train:
        check_env()
        if FLAGS.clean:
            clean()
        if os.path.exists('config_file'):
            with open('config_file', 'r', encoding='utf-8') as r:
                config = json.load(r)
        else:
            config = get_config()

        train(config, logger)
    else:
        if os.path.exists('config_file'):
            with open('config_file', 'r', encoding='utf-8') as r:
                config = json.load(r)
        else:
            raise FileNotFoundError('can not find config_file')

        test(config, logger)


if __name__ == '__main__':
    tf.app.run(main)

