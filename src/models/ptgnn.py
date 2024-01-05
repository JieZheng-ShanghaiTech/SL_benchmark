import tensorflow as tf
import numpy as np

# random.seed(123)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, n_node, input_dim, output_dim, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = "Convolution"
        self.vars = {}
        self.issparse = False
        self.n_node = n_node
        with tf.compat.v1.variable_scope(self.name):
            self.vars['weights1'] = glorot([input_dim, output_dim])
            self.vars['weights2'] = weight_variable_glorot(output_dim, 64, name='weights2')

        self.dropout = dropout
        self.act = act
        self.issparse = True
        self.norm = norm
        self.is_train = is_train

    def encoder(self, inputs, adj):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.matmul(x, self.vars['weights1'])
            x = tf.sparse_tensor_dense_matmul(adj, x)
            outputs = self.act(x)

            x2 = tf.matmul(outputs, self.vars['weights2'])
            x2 = tf.sparse_tensor_dense_matmul(adj, x2)
            outputs = self.act(x2)

        if self.norm:
            outputs = tf.layers.batch_normalization(outputs, training=self.is_train)
        return outputs

    def decoder(self, embed, nd):
        embed_size = embed.shape[1].value
        logits = tf.matmul(embed, tf.transpose(embed))
        logits = tf.reshape(logits, [-1, 1])
        # return logits
        return tf.nn.relu(logits)

    def training(self, loss, lr, l2_coef):
        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss)
        return train_op


def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds - labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
    #     return tf.reduce_sum(error)
    return tf.sqrt(tf.reduce_mean(error))


def ROC(outs, labels, test_arr, label_neg):
    scores = []
    for i in range(len(test_arr)):
        l = test_arr[i]
        scores.append(outs[int(labels[l, 0] - 1), int(labels[l, 1] - 1)])

    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i, 0]), int(label_neg[i, 1])])

    test_labels = np.ones((len(test_arr), 1))
    temp = np.zeros((label_neg.shape[0], 1))
    test_labels1 = np.vstack((test_labels, temp))
    test_labels1 = np.array(test_labels1, dtype=np.bool).reshape([-1, 1])
    return test_labels1, scores

class Model:
    def __init__(self, do_train=True):
        self.batch_size = 1
        self.batch_num = 64
        self.lr = 0.005
        self.l2_coef = 0.0005
        self.weight_decay = 5e-3
        self.nonlinearity = tf.nn.relu
        self.token_size = 9724
        # self.token_size = 8001
        self.dim_embedding = 100
        # self.len_sequence = 800
        self.len_sequence = 600
        self.num_filter = 1
        self.do_train = do_train
        if self.do_train:
            self.num_nodes = 20398
            # self.num_nodes = 20375
        else:
            self.num_nodes = 9845
            # self.num_nodes = 6375
        self.entry_size = self.num_nodes ** 2

        if self.do_train:
            with tf.name_scope('input_train'):
                self.encoded_protein = tf.compat.v1.placeholder(dtype=tf.float32,
                                                                shape=(self.num_nodes, self.len_sequence))
                self.bias_in1 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph_train1')
                self.bias_in2 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph_train2')
                self.lbl_in1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.lbl_in2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.msk_in1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.msk_in2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.neg_msk1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.neg_msk2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
        else:
            with tf.name_scope('input_train'):
                self.encoded_protein = tf.compat.v1.placeholder(dtype=tf.float32,
                                                                shape=(self.num_nodes, self.len_sequence))
                self.bias_in = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph_train')
                self.lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))

        self.instantiate_embeddings()
        self.logits = self.inference()
        self.loss, self.accuracy = self.loss_func()
        self.train_op = self.train()

    def inference(self):
        embedding_proteins = self.word2sentence()
        embedding_proteins = tf.nn.l2_normalize(embedding_proteins, 1)

        self.model = GraphConvolutionSparse(
            n_node=embedding_proteins.shape[0].value,
            input_dim=embedding_proteins.shape[1].value,
            output_dim=128,
            act=tf.nn.leaky_relu,
            dropout=0.25)
        if self.do_train:
            self.final_embedding1 = self.model.encoder(embedding_proteins, self.bias_in1)
            self.final_embedding2 = self.model.encoder(embedding_proteins, self.bias_in2)
            self.logits1 = self.model.decoder(self.final_embedding1, self.num_nodes)
            self.logits2 = self.model.decoder(self.final_embedding2, self.num_nodes)
            logits = (self.logits1 + self.logits2) / 2
            self.final_embedding = (self.final_embedding1 + self.final_embedding2) / 2
            return logits
        else:
            self.final_embedding = self.model.encoder(embedding_proteins, self.bias_in)
            print(embedding_proteins.shape)
            print(self.final_embedding.shape)
            logits = self.model.decoder(self.final_embedding, self.num_nodes)
            return logits

    def loss_func(self):
        if self.do_train:
            loss1 = masked_accuracy(self.logits1, self.lbl_in1, self.msk_in1, self.neg_msk1)
            loss2 = masked_accuracy(self.logits2, self.lbl_in2, self.msk_in2, self.neg_msk2)
            loss = (loss1 + loss2) / 2
            accuracy = loss
        else:
            loss = masked_accuracy(self.logits, self.lbl_in, self.msk_in, self.neg_msk)
            accuracy = self.logits
        return loss, accuracy

    def train(self):
        train_op = self.model.training(self.loss, self.lr, self.l2_coef)
        return train_op

    ############  --- convolution ---  #################
    def word2sentence(self):
        return self.conv1dim(tf.cast(self.encoded_protein, dtype=tf.int32))

    def conv1dim(self, batch_protein):
        embedding_protein = tf.nn.embedding_lookup(self.embedding_tokens, batch_protein)
        embedding_protein = tf.layers.conv1d(embedding_protein, 16, 10, use_bias=True, padding="valid",
                                             activation='relu')
        embedding_protein = tf.layers.max_pooling1d(embedding_protein, pool_size=60, strides=60)
        final_embedding_protein = tf.contrib.layers.flatten(embedding_protein)
        return final_embedding_protein

    def instantiate_embeddings(self):
        """define all embeddings here"""
        if self.do_train:
            with tf.name_scope("token_embedding"):
                self.embedding_tokens = tf.compat.v1.get_variable("embedding", shape=[self.token_size, self.dim_embedding],
                                                        initializer=tf.random_normal_initializer(stddev=0.1))
        else:
            init_emb=np.load('../data/preprocessed_data/ptgnn_data/trained_word_embedding.npy')
            with tf.name_scope("token_embedding"):
                self.embedding_tokens = tf.compat.v1.get_variable("embedding", initializer=init_emb)


