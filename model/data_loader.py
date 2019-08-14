import numpy as np
from tqdm import tqdm
import random
import re
from model.util import *


class data_loader(object):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.sel_relation = set(config['sel_label']) if config['sel_label'] else None

    def load(self, use_small=False):
        dataset_ret = []
        pbar = tqdm(total=len(self.dataset) if not use_small else 100)
        badcase = 0

        for i, data in enumerate(self.dataset):
            if use_small and i == 100:
                print('use_samll', i)
                break

            sentText = data['sentText']
            sentText = re.sub('\s+', ' ', sentText) # 删除多于空格
            relation = data['relationMentions']
            entitys = data['entityMentions']

            # 构建entity到type的映射表
            entity_to_type = {}
            for entity in entitys:
                mentions = '<START>{}<END>'.format(entity['text']).split(' ')
                for mention in mentions:
                    entity_to_type[mention] = entity['label']

            for rel in relation:
                if self.sel_relation and rel['label'] not in self.sel_relation:
                    continue

                entity1 = rel['em1Text']
                entity2 = rel['em2Text']
                label = conver_token_to_id(rel['label'], self.config['label_dict'])


                # todo: 用之前的entity1 pattern entity2 的方法找， 然后去位置考前的
                ent1_ent2_search = re.search(r'(?: |^)({0} (.*?) {1})(?: |$)'.format(entity1, entity2), sentText)
                ent2_ent1_search = re.search(r'(?: |^)({1} (.*?) {0})(?: |$)'.format(entity1, entity2), sentText)
                if ent1_ent2_search and ent2_ent1_search:
                    find = ent1_ent2_search if ent1_ent2_search.start() < ent2_ent1_search.start() else ent2_ent1_search
                    flag = 0 if ent1_ent2_search.start() < ent2_ent1_search.start() else 1
                elif ent1_ent2_search:
                    find = ent1_ent2_search
                    flag = 0
                elif ent2_ent1_search:
                    find = ent2_ent1_search
                    flag = 1
                else:
                    badcase += 1
                    continue
                pattern = find.group(2)
                start, end = find.span(1)

                if flag == 0:
                    short = re.sub(r'^{} '.format(entity1), '<START>{}<END> '.format(entity1), find.group(1))
                    short = re.sub(r' {}$'.format(entity2), ' <START>{}<END>'.format(entity2), short)
                elif flag == 1:
                    short = re.sub(r'^{} '.format(entity2), '<START>{}<END> '.format(entity2), find.group(1))
                    short = re.sub(r' {}$'.format(entity1), ' <START>{}<END>'.format(entity1), short)
                sentence = sentText[:start] + short + sentText[end:]
                sentence = sentence.split(' ')

                positions = get_positions(sentence, flag, self.config['pos_max'])
                types = get_types(sentence, entity_to_type, self.config['type_dict'])

                # 构建每个词的id 和 att_label
                feature = []
                att_label = [0.] * len(sentence)  # 用于attetion regulation
                att_flag = 0
                for i, word in enumerate(sentence):
                    if word[:7] == '<START>':
                        word = word[7:]
                        att_label[i] = 1.
                        att_flag += 1
                    if word[-5:] == '<END>':
                        word = word[:-5]
                        att_label[i] = 1.
                        att_flag += 1
                    if att_flag > 1 and att_flag < 4:
                        att_label[i] = 1.
                    wordid = conver_token_to_id(word, self.config['vocab'])
                    feature.append(wordid)
                fenmu = sum(att_label)
                att_label = [f / fenmu for f in att_label]

                dataset_ret.append([feature, positions, types, att_label, label, pattern])
            pbar.update(1)
        pbar.close()
        self.dataset = dataset_ret
        return badcase

    def load_v2(self, use_small=None):
        dataset_ret = []
        pbar = tqdm(total=len(self.dataset) if not use_small else use_small)

        for i, data in enumerate(self.dataset):
            if use_small and i == use_small:
                print('use_samll', i)
                break

            sentText = data['sentText']
            relation = data['relationMentions']
            entitys = data['entityMentions']

            # 构建entity到type的映射表
            entity_to_type = {}
            for entity in entitys:
                mentions = '<START>{}<END>'.format(entity['text']).split(' ')
                for mention in mentions:
                    entity_to_type[mention] = entity['label']

            for rel in relation:
                entity1 = rel['em1Text']
                entity2 = rel['em2Text']
                label = conver_token_to_id(rel['label'], self.config['label_dict'])

                # 对于每一对entity pair, 在句子中存在entity1(.*?)entity2 和 entity2(.*?)entity1两种情况
                # 利用正则表达式找出以上的所有情况
                findalls = []
                findall = re.finditer(r'(?: |^)({0}( .*? ){1})(?: |$)'.format(entity1, entity2), sentText)
                for find in findall:
                    if find:
                        findalls.append((find, 0))
                findall = re.finditer(r'(?: |^)({1}( .*? ){0})(?: |$)'.format(entity1, entity2), sentText)
                for find in findall:
                    if find:
                        findalls.append((find, 1))

                # 对上面找出的所有情况构建训练数据
                for find, flag in findalls:
                    start, end = find.span(1)
                    if flag == 0:
                        short = re.sub(r'^{} '.format(entity1), '<START>{}<END> '.format(entity1), find.group(1))
                        short = re.sub(r' {}$'.format(entity2), ' <START>{}<END>'.format(entity2), short)
                    elif flag == 1:
                        short = re.sub(r'^{} '.format(entity2), '<START>{}<END> '.format(entity2), find.group(1))
                        short = re.sub(r' {}$'.format(entity1), ' <START>{}<END>'.format(entity1), short)
                    sentence = sentText[:start] + short + sentText[end:]
                    pattern = find.group(2)
                    sentence = sentence.split(' ')

                    positions = get_positions(sentence, flag, self.config['pos_max'])
                    types = get_types(sentence, entity_to_type, self.config['type_dict'])

                    # 构建每个词的id 和 att_label
                    feature = []
                    att_label = [0.] * len(sentence)  # 用于attetion regulation
                    att_flag = 0
                    for i, word in enumerate(sentence):
                        if word[:7] == '<START>':
                            word = word[7:]
                            att_label[i] = 1.
                            att_flag += 1
                        if word[-5:] == '<END>':
                            word = word[:-5]
                            att_label[i] = 1.
                            att_flag += 1
                        if att_flag >1 and att_flag<4:
                            att_label[i] = 1.
                        wordid = conver_token_to_id(word, self.config['vocab'])
                        feature.append(wordid)
                    fenmu = sum(att_label)
                    att_label = [f / fenmu for f in att_label]

                    dataset_ret.append([feature, positions, types, att_label, label, pattern])
            pbar.update(1)
        pbar.close()

        print('load data: {}'.format(len(dataset_ret)))
        self.dataset = dataset_ret

    def minibatch(self, batch_size, shuffle=False, redistribution=False, trustable_pattern=None):
        idxs = list(range(len(self.dataset)))
        if shuffle:
            random.shuffle(idxs)

        features = []
        position1s = []
        position2s = []
        typess = []
        labels = []
        att_labels = []
        lengths = []
        patterns = []

        for idx in idxs:
            feature, positions, types, att_label, label, pattern = self.dataset[idx]
            features.append(feature)
            lengths.append(len(feature))
            position1s.append(positions[0])
            position2s.append(positions[1])
            typess.append(types)
            if redistribution and pattern not in trustable_pattern[label]:
                label = conver_token_to_id('None', self.config['label_dict'])
            labels.append(label)
            att_labels.append(att_label)
            patterns.append(pattern)

            if len(features) % batch_size == 0:
                features = np.array(padding(features))
                position1s = np.array(padding(position1s))
                position2s = np.array(padding(position2s))
                typess = np.array(padding(typess))
                att_labels = np.array(padding(att_labels))
                lengths = np.array(lengths)
                att_labels = np.array(att_labels)

                yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels
                features = []
                position1s = []
                position2s = []
                typess = []
                labels = []
                att_labels = []
                lengths = []
                patterns = []

        if len(features) > 0:
            features = np.array(padding(features))
            position1s = np.array(padding(position1s))
            position2s = np.array(padding(position2s))
            typess = np.array(padding(typess))
            att_labels = np.array(padding(att_labels))
            lengths = np.array(lengths)
            att_labels = np.array(att_labels)

            yield patterns, features, position1s, position2s, typess, lengths, att_labels, labels

    def __len__(self):
        return len(self.dataset)
















