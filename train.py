import tensorflow as tf
from model.model import model
from model.data_loader import data_loader
from model.util import load_data_file, get_vocab, get_typeDict, get_labelDict
import os
import shutil
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_config():
    config = {}


    config['batchsize'] = 256

    config['word_dim'] = 100
    config['position_dim'] = 50
    config['type_dim'] = 50
    config['hidden_dim'] = 500

    config['pos_max'] = 60

    config['redistribution'] = True
    config['attention_regularization'] = True
    config['bootstrap'] = True
    config['zero'] = True
    config['lower'] = True

    config['first_loop_epoch'] = 10
    config['epoch'] = 100

    config['init_patterns_ratio'] = 0.1
    config['init_patterns_max'] = 20

    config['pattern_threshold'] = 0.5
    config['pattern_max'] = 5

    config['beta'] = 1.
    config['lr'] = 0.001
    config['clip'] = 5.
    config['lr_method'] = 'adam'

    config['restore'] = False

    return config

def get_logger():
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename="train.log")

    logger.setLevel(logging.INFO)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# chech env
def check_env():
    if not os.path.exists('./ckpt_model'):
        os.mkdir('./ckpt_model')
    if not os.path.exists('./best_model'):
        os.mkdir('./best_model')



def parser_score(epoch, best_score, cur_score, accs, config, logger, mode='val'):
    flag = False
    if mode.lower() != 'test' and cur_score['all']['F1'] > best_score[0]:
        if not config['booststrap']:
            for file in os.listdir('./ckpt_model/'):
                filename = os.path.join('./ckpt_model/', file)
                shutil.copy(filename, 'best_model/')

        best_score[0] = cur_score['all']['F1']
        best_score[1] = epoch
        best_score[2] = cur_score
        best_score[3] = accs
        flag = True

    logger.info('epoch: {}'.format(epoch))
    logger.info('acc: {}'.format(accs))

    for k, v in cur_score.items():
        logger.info('{:<45d}:\tP: {:>.2f}\tR: {:>.2f}\tF1: {:>.2f}'.format(k, v['P'] * 100, v['R'] * 100, v['F1'] * 100))
    if mode.lower() != 'test':
        logger.info('best_F1:{:^.2f} in epoch:{}'.format(best_score[0] * 100, best_score[1]))

    return best_score, flag



def main():
    logger = get_logger()
    config = get_config()

    train_datas = load_data_file('datas/train.json', zero=config['zero'], lower=config['lower'])
    vocab = get_vocab([train_datas])
    type_dict = get_typeDict([train_datas])
    label_dict = get_labelDict([train_datas])

    config['word_num'] = len(vocab)
    config['type_num'] = len(type_dict)
    config['label_num'] = len(label_dict)
    config['position_num'] = max([len(d['sentText'].split(' ')) for d in train_datas]) if config['pos_max'] < 0 else config['pos_max']

    for k, v in config.items():
        logger.info('config {}: {}'.format(k, v))

    config['vocab'] = vocab
    config['type_dict'] = type_dict
    config['label_dict'] = label_dict


    train_data_loader = data_loader(train_datas, config)
    train_data_loader.load()

    val_datas = load_data_file('datas/dev.json', zero=config['zero'], lower=config['lower'])
    val_data_loader = data_loader(val_datas, config)
    val_data_loader.load()

    test_datas = load_data_file('datas/test.json', zero=config['zero'], lower=config['lower'])
    test_data_loader = data_loader(test_datas, config)
    test_data_loader.load()



    pattern_dict = {}
    for data in train_data_loader.dataset:
        pattern = data[-1]
        label = data[-2]
        if label not in pattern_dict:
            pattern_dict[label] = {}
        if pattern not in pattern_dict[label]:
            pattern_dict[label][pattern] = 0
        pattern_dict[label][pattern] += 1

    if config['booststrap'] or config['redistribution']:
        logger.info('init patterns')
        trustable_pattern = {}
        for k, v in pattern_dict.items():
            v = sorted(v.items(), key=lambda x: x[1], reverse=True)
            sel_num = int(config['init_patterns_ratio'] * len(v))
            sel_num =  config['init_patterns_max'] if sel_num> config['init_patterns_max'] else sel_num
            trustable_pattern[k] = set(v1 for v1, v2 in v[:sel_num])
        config['trustable_pattern'] = trustable_pattern


    with tf.Graph().as_default() as g:
        train_model = model(config, 'train')
        train_model.build(g)

    with tf.Graph().as_default() as g:
        val_model = model(config, 'val')
        val_model.build(g)

    best_score = [-1., 0, None, None]

    # for not boostrap or firt loop of bootstrap
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
        best_score, is_new = parser_score(epoch, best_score, cur_score, accs, config, logger)
        if is_new:
            logger.info('***TEST***')
            test_score, test_accs = val_model.run_evaule(test_data_loader)
            parser_score(epoch, '', test_score, test_accs, config, logger, 'test')
        logger.info('******\n')


    # for other bootstrap loop
    if config['bootstrap']:

        # update patterns
        pattern_condidates = {}
        for kl, pattern, label in zip(kls, patterns, labels):
            pattern_score = 1 / (1 + kl)
            if pattern_score>config['pattern_threshold']:
                if label not in pattern_condidates:
                    pattern_condidates[label] = []
                pattern_condidates[label].append((pattern, pattern_score))

        for label in labels:
            if label not in pattern_condidates:
                continue
            pattern_condidates[label] = sorted(pattern_condidates[label], key=lambda x: x[1], reverse=True)
            count = 0
            for condidate in pattern_condidates[label]:
                if condidate not in config['trustable_pattern'][label]:
                    config['trustable_pattern'][label].add(condidate)
                    count+=1
                if count >= config['pattern_max'] :
                    break
        train_model.update_trustable_pattern(config)

        for epoch in range(config['epoch'] - config['first_loop_epoch']):
            logger.info('***TRAIN: {}***'.format(epoch))

            if config['attention_regularization']:
                attentions, kls, patterns, labels = train_model.run_train(train_data_loader)
            else:
                patterns, labels = train_model.run_train(train_data_loader)

            logger.info('***VAL***')
            cur_score, accs = val_model.run_evaule(val_data_loader)
            best_score, is_new = parser_score(epoch, best_score, cur_score, accs, config, logger)
            if is_new:
                logger.info('***TEST***')
                test_score, test_accs = val_model.run_evaule(test_data_loader)
                parser_score(epoch, '', test_score, test_accs, config, logger, 'test')
            logger.info('******\n')

            # update patterns
            pattern_condidates = {}
            for kl, pattern, label in zip(kls, patterns, labels):
                pattern_score = 1 / (1 + kl)
                if pattern_score>config['pattern_threshold']:
                    if label not in pattern_condidates:
                        pattern_condidates[label] = []
                    pattern_condidates[label].append((pattern, pattern_score))

            for label in labels:
                if label not in pattern_condidates:
                    continue
                pattern_condidates[label] = sorted(pattern_condidates[label], key=lambda x: x[1], reverse=True)
                count = 0
                for condidate in pattern_condidates[label]:
                    if condidate not in config['trustable_pattern'][label]:
                        config['trustable_pattern'][label].add(condidate)
                        count+=1
                    if count >= config['pattern_max'] :
                        break
            train_model.update_trustable_pattern(config)


if __name__ == '__main__':
    main()


