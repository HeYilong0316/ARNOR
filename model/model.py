import tensorflow as tf
from tqdm import tqdm
import numpy as np




class model(object):
    def __init__(self, config, mode):
        self.config = config
        self.lr = config['lr'] if 'lr' in config else None
        self.id_to_label = {v: k for k, v in self.config['label_dict'].items()}
        self.mode = mode

    def placeholder_op(self):
        
        self.word_ids = tf.placeholder(name='word_ids',
                                       shape=[None, None],
                                       dtype=tf.int32)

        self.position1_ids = tf.placeholder(name='position1_ids', 
                                            shape=[None, None],
                                            dtype=tf.int32)

        self.position2_ids = tf.placeholder(name='position2_ids',
                                            shape=[None, None],
                                            dtype=tf.int32)

        self.type_ids = tf.placeholder(name='type_ids',
                                       shape=[None, None],
                                       dtype=tf.int32)

        self.att_labels = tf.placeholder(name='att_label',
                                        shape = [None, None],
                                        dtype=tf.float32)
        
        self.lengths = tf.placeholder(name='lengths',
                                     shape=[None],
                                     dtype=tf.int32)

        self.label_ids = tf.placeholder(name='label_ids',
                                        shape=[None],
                                        dtype=tf.int32)

        self.learn_rate = tf.placeholder(name='lr',
                                         shape=[],
                                         dtype=tf.float32)




    def forward(self):
        
        with tf.variable_scope('embedding'):
            word_embedding_matrix = tf.get_variable(name='word_embedding_matrix',
                                             shape=[self.config['word_num'], self.config['word_dim']],
                                             dtype=tf.float32
                                             )

            pos1_embedding_matrix = tf.get_variable(name='pos1_embedding_matrix',
                                             shape=[self.config['position_num'], self.config['position_dim']],
                                             dtype=tf.float32
                                             )

            pos2_embedding_matrix = tf.get_variable(name='pos2_embedding_matrix',
                                             shape=[self.config['position_num'], self.config['position_dim']],
                                             dtype=tf.float32
                                             )

            type_embedding_matrix = tf.get_variable(name='type_embedding_matrix',
                                             shape=[self.config['type_num'], self.config['type_dim']],
                                             dtype=tf.float32
                                             )

            word_embeddings = tf.nn.embedding_lookup(params=word_embedding_matrix,
                                                     ids=self.word_ids,
                                                     name='word_embeddings_look_up')

            pos1_embeddings = tf.nn.embedding_lookup(params=pos1_embedding_matrix,
                                                     ids=self.position1_ids,
                                                     name='pos1_embeddings_look_up')

            pos2_embeddings = tf.nn.embedding_lookup(params=pos2_embedding_matrix,
                                                     ids=self.position2_ids,
                                                     name='pos2_embeddings_look_up')

            type_embeddings = tf.nn.embedding_lookup(params=type_embedding_matrix,
                                                     ids=self.type_ids,
                                                     name='type_embeddings_look_up')

            bilstm_input = tf.concat([word_embeddings, pos1_embeddings, pos2_embeddings, type_embeddings], -1)

        with tf.variable_scope('BiLSTM_ATT'):
            lstm_cell_forward = tf.keras.layers.LSTMCell(self.config['hidden_dim']//2, name='lstm_cell_forward')
            lstm_cell_backward = tf.keras.layers.LSTMCell(self.config['hidden_dim']//2, name='lstm_cell_backward')

            # (B, max_length, hidden_dim)
            lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_forward,
                                                             cell_bw=lstm_cell_backward,
                                                             inputs=bilstm_input,
                                                             sequence_length=self.lengths,
                                                             dtype=tf.float32
                                                        )
            hiddens = tf.concat(lstm_output, -1)

            # (B, max_length, hidden_dim)
            hiddens_for_att =  tf.squeeze(tf.keras.layers.Dense(1)(hiddens), -1)
            # (B, max_length)
            self.attentions = self.attention_fun(hiddens_for_att, self.lengths)

            # (B, hidden_dim)
            hidden_att = tf.reduce_sum(hiddens * tf.expand_dims(self.attentions, -1), 1)

            self.logits = tf.keras.layers.Dense(self.config['label_num'])(hidden_att)



    @staticmethod
    def attention_fun(hiddens, lengths):
        batchsize = tf.shape(hiddens)[0]
        max_length = tf.reduce_max(lengths)

        def __attention__(bid):
            # (1)
            length = lengths[bid]
            # (length)
            attention_logit = hiddens[bid][:length]
            # (length)
            attention = tf.nn.softmax(attention_logit)
            # (max_length), (a_1,a_2...a_length,0,0,0...)
            return tf.concat([attention, tf.zeros(max_length-length)], -1)

        # [(max_length), (max_length)...] -> [batchsize, max_length]
        return tf.concat(tf.map_fn( __attention__, tf.range(batchsize), dtype=tf.float32), 0)



    def loss_op(self):

        with tf.variable_scope('loss'):
            loss_c = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ids, logits=self.logits)
            attention = tf.clip_by_value(self.attentions, 1e-10, 1)

            if self.config['attention_regularization']:
                loss_a = tf.keras.losses.kullback_leibler_divergence(y_true=self.att_labels, y_pred=attention)
                self.kl = loss_a
                self.loss = tf.reduce_mean(loss_c + self.config['beta'] * loss_a)
            else:
                self.loss = tf.reduce_mean(loss_c)


    def add_train_op(self):
        """Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping
        """
        _lr_m = self.config['lr_method'].lower() # lower to make sure
        lr = self.learn_rate
        clip = self.config['clip']
        loss = self.loss

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def predict(self, data_batch):
        feed = self.get_feed(data_batch)
        data_score = self.sess.run(self.logits, feed)
        data_pre = np.argmax(data_score, -1)
        return data_pre
        
        
    def Session(self, graph):

        self.sess = tf.Session(graph=graph)
        self.saver = tf.train.Saver()

        if self.mode.lower() == 'train' and not self.config['restore']:
            print('random init')
            self.sess.run(tf.global_variables_initializer())
        elif self.mode.lower() == 'train':
            print('restore model from ckpt_model for train')
            self.saver.restore(self.sess, './ckpt_model/')


    def build(self, graph):
        self.placeholder_op()
        self.forward()
        if self.mode.lower() == 'train':
            self.loss_op()
            self.add_train_op()
        self.Session(graph)


    def get_feed(self, datas):

        if len(datas)>5: # for train
            assert len(datas)==7,  [len(datas)]
            features, position1s, position2s, typess, att_labels, lengths, labels = datas
        else: # for predict
            assert len(datas)==5, [len(datas)]
            features, position1s, position2s, typess, lengths = datas
            att_labels, labels = None, None

        feed = {self.word_ids: features,
                self.position1_ids: position1s,
                self.position2_ids: position2s,
                self.type_ids: typess,
                self.lengths:lengths
                }

        if self.lr is not None:
            feed[self.learn_rate] = self.lr

        if att_labels is not None:
            feed[self.att_labels] = att_labels

        if labels is not None:
            feed[self.label_ids] = labels

        return feed


    def run_train(self, train_data):
        all_iter = len(train_data)//self.config['batchsize']
        if len(train_data) % self.config['batchsize']!=0:
            all_iter += 1

        train_data_iter = train_data.minibatch(self.config['batchsize'],
                                               shuffle=True,
                                               redistribution=self.config['redistribution'],
                                               trustable_pattern=self.config['trustable_pattern'])

        pbar = tqdm(total=all_iter)

        attentions = []
        kls = []
        patternss = []
        labelss = []

        for patterns, features, position1s, position2s, typess, lengths, att_labels, labels in train_data_iter:
            feed = self.get_feed([features, position1s, position2s, typess, att_labels, lengths, labels])
            if self.config['attention_regularization']:
                _, loss_batch, attention, kl= self.sess.run([self.train_op, self.loss, self.attentions, self.kl], feed)
                attentions.extend(attention)
                kls.extend(kl)
            else:
                _, loss_batch = self.sess.run([self.train_op, self.loss], feed)
            patternss.extend(patterns)
            labelss.extend(labels)
            pbar.set_description('loss: {}'.format(loss_batch))
            pbar.update(1)
        pbar.close()
        self.saver.save(self.sess, './ckpt_model/')
        if self.config['attention_regularization']:
            return attentions, kls, patternss, labelss
        else:
            return patternss, labelss


    def run_evaule(self, val_data):
        model_dir = './ckpt_model/'
        model_dir = './best_model/' if self.mode == 'val2' else model_dir
        print('restore model from {} for val or test'.format(model_dir))
        self.saver.restore(self.sess, model_dir)

        all_iter_val = len(val_data)//self.config['batchsize']
        if len(val_data) % self.config['batchsize']!=0:
            all_iter_val +=1

        val_data_iter = val_data.minibatch(self.config['batchsize'])
        predict_val_epoch = []
        label_val_epoch = []

        pbar = tqdm(total=all_iter_val)
        for patterns, features, position1s, position2s, typess, lengths, att_labels, labels in val_data_iter:
            predict_val_batch = self.predict([features, position1s, position2s, typess, lengths])
            predict_val_epoch.extend(predict_val_batch)
            label_val_epoch.extend(labels)
            pbar.update(1)
        pbar.close()

        accs = np.mean([int(p==l) for p, l in zip(predict_val_epoch, label_val_epoch)])
        cur_score = {}

        for i in range(self.config['label_num']):
            i = self.id_to_label[i]
            pre_true_count = 0
            lab_count = 0
            pre_count = 0
            for pre, lab in zip(predict_val_epoch, label_val_epoch):
                pre = self.id_to_label[pre]
                lab = self.id_to_label[lab]
                if pre == i:
                    pre_count += 1
                if lab == i:
                    lab_count += 1
                if pre == i and lab == i:
                    pre_true_count += 1

            P = (pre_true_count / pre_count) if pre_count > 0 else 0
            R = (pre_true_count / lab_count) if lab_count > 0 else 0
            F = (2 * (P * R) / (P + R)) if (P + R) > 0 else 0
            cur_score[str(i)] = {'P': P, 'R': R, 'F1': F}

        pre_true_count = 0
        lab_count = 0
        pre_count = 0
        for pre, lab in zip(predict_val_epoch, label_val_epoch):
            pre = self.id_to_label[pre]
            lab = self.id_to_label[lab]

            if pre != 'None':
                pre_count += 1
            if lab != 'None':
                lab_count += 1
            if pre == lab and pre != 'None':
                pre_true_count += 1
        P = (pre_true_count / pre_count) if pre_count > 0 else 0
        R = (pre_true_count / lab_count) if lab_count > 0 else 0
        F = (2 * (P * R) / (P + R)) if (P + R) > 0 else 0
        cur_score['all'] = {'P': P, 'R': R, 'F1': F}

        return cur_score, accs


    def update_trustable_pattern(self, config):
        self.config['trustable_pattern'] = config['trustable_pattern']
