import tensorflow as tf
from abc import abstractmethod
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import sklearn.metrics as m
import pandas as pd
import copy
import numpy as np

LAYER_IDS = {}

# random.seed(123)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)

"""
class Aggregator and SumAggregator refer to http://arxiv.org/abs/1905.04413 and https://dl.acm.org/doi/10.1145/3308558.3313417.
"""
tf.keras.layers.MaxPool1D

def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # nodea_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, nodea_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            nodea_embeddings = tf.reshape(nodea_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            nodea_relation_scores = tf.reduce_mean(nodea_embeddings * neighbor_relations, axis=-1)
            nodea_relation_scores_normalized = tf.nn.softmax(nodea_relation_scores, axis=-1)

            # [batch_size, -1, n_neighbor, 1]
            nodea_relation_scores_normalized = tf.expand_dims(nodea_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(nodea_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.compat.v1.variable_scope(self.name):
            self.weights = tf.compat.v1.get_variable(shape=[self.dim, self.dim],
                                           initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.compat.v1.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, nodea_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, nodea_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, rate=self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


"""
class KG4SL refers to http://arxiv.org/abs/1905.04413 and https://dl.acm.org/doi/10.1145/3308558.3313417.
"""


class KG4SL(object):
    def __init__(self, n_hop, batch_size, neighbor_sample_size, dim, l2_weight, lr, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(n_hop, batch_size, neighbor_sample_size, dim, l2_weight, lr, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, n_hop, batch_size, neighbor_sample_size, dim, l2_weight, lr, adj_entity, adj_relation):
        self.adj_entity = adj_entity  # [entity_num, neighbor_sample_size]
        self.adj_relation = adj_relation
        self.n_hop = n_hop
        self.batch_size = batch_size
        self.n_neighbor = neighbor_sample_size
        self.dim = dim
        self.l2_weight = l2_weight
        self.lr = lr

    def _build_inputs(self):
        self.nodea_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None], name='nodea_indices')
        self.nodeb_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None], name='nodeb_indices')
        self.labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_entity, n_relation):
        self.entity_emb_matrix = tf.compat.v1.get_variable(shape=[n_entity, self.dim], initializer=KG4SL.get_initializer(),name='entity_emb_matrix')
        self.relation_emb_matrix = tf.compat.v1.get_variable(shape=[n_relation, self.dim], initializer=KG4SL.get_initializer(),name='relation_emb_matrix')

        # [batch_size, dim]
        nodea_embeddings_initial = tf.nn.embedding_lookup(self.entity_emb_matrix, self.nodea_indices)
        nodeb_embeddings_initial = tf.nn.embedding_lookup(self.entity_emb_matrix, self.nodeb_indices)

        nodea_entities, nodea_relations = self.get_neighbors(self.nodea_indices)
        nodeb_entities, nodeb_relations = self.get_neighbors(self.nodeb_indices)

        # [batch_size, dim]
        self.nodea_embeddings, self.nodea_aggregators = self.aggregate(nodea_entities, nodea_relations, nodeb_embeddings_initial)
        self.nodeb_embeddings, self.nodeb_aggregators = self.aggregate(nodeb_entities, nodeb_relations, nodea_embeddings_initial)

        # [batch_size]
        self.scores = tf.reduce_sum(self.nodea_embeddings * self.nodeb_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_hop):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    # feature propagation
    def aggregate(self, entities, relations, embeddings_agg):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        embeddings_aggregator = embeddings_agg

        for i in range(self.n_hop):
            if i == self.n_hop - 1:
                aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_hop - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    nodea_embeddings=embeddings_aggregator,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    # loss
    def _build_train(self):
        # mse loss
        self.mse_loss = tf.reduce_mean(tf.square(self.labels-self.scores))

        # base loss
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # L2 loss
        self.l2_loss = tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)

        for aggregator in self.nodeb_aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        for aggregator in self.nodea_aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        # changed loss
        # self.loss = self.mse_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        nodea_emb, nodeb_emb = sess.run([self.nodea_embeddings, self.nodeb_embeddings], feed_dict)
        scores_output = copy.deepcopy(scores)

        auc = roc_auc_score(y_true=labels, y_score=scores)
        p, r, t = precision_recall_curve(y_true=labels, probas_pred=scores)
        aupr = m.auc(r, p)

        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        scores_binary_output = scores

        f1 = f1_score(y_true=labels, y_pred=scores)
        return nodea_emb, nodeb_emb, scores_output, scores_binary_output, auc, f1, aupr

    def cal_scores(self, sess, feed_dict):
        scores = sess.run(self.scores_normalized, feed_dict)

        return scores

    def get_scores(self, sess, feed_dict):
        return sess.run([self.nodeb_indices, self.scores_normalized], feed_dict)
