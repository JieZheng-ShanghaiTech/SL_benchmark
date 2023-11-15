import numpy as np
import tensorflow as tf

# flags = tf.app.flags
# FLAGS = flags.FLAGS

# random.seed(123)
np.random.seed(456)
tf.compat.v1.set_random_seed(456)

# torch.manual_seed(123)
# torch.cuda.manual_seed_all(123)

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.norm = norm
        self.is_train = is_train

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse.sparse_dense_matmul(self.adj, x)
            outputs = self.act(x)
            if self.norm:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        self.norm = norm
        self.is_train = is_train

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse.sparse_dense_matmul(x, self.vars['weights'])
            x = tf.sparse.sparse_dense_matmul(self.adj, x)
            outputs = self.act(x)
            if self.norm:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

        return outputs


class AttentionRec():
    """Attention merge layer for each support view"""

    def __init__(self, output_dim, num_support, name, dropout=0., act=tf.nn.sigmoid):
        self.num_nodes = output_dim
        self.num_support = num_support
        self.name = name
        self.dropout = dropout
        self.act = act
        self.attADJ = []
        with tf.compat.v1.variable_scope(self.name + '_attW'):
            self.attweights = tf.compat.v1.get_variable("attWeights", [self.num_support, self.num_nodes, self.num_nodes],
                                             initializer=tf.random_uniform_initializer(minval=0.9, maxval=1.1))
            self.attention = tf.nn.softmax(self.attweights, 0)


    def __call__(self, recs):
        with tf.name_scope(self.name):
            for i in range(self.num_support):
                self.attADJ.append(tf.multiply(self.attention[i], recs[i]))
            confiWeights = tf.add_n(self.attADJ)

            return confiWeights


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, output_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.vars = {}
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(output_dim, output_dim, name='weights')
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            x = tf.matmul(inputs, self.vars['weights'])
            x = tf.matmul(x, tf.transpose(inputs))
            outputs = self.act(x)
        return outputs

class Optimizer():
    def __init__(self, supp, main, preds, labels, Alpha, Beta, learning_rate, num_nodes, num_edges, index):
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            labels_sub = tf.gather_nd(labels, index)
            main_sub = tf.gather_nd(main, index)
            preds_sub = tf.gather_nd(preds, index)

            self.loss_supp = 0
            for viewRec in supp:
                viewRec_sub = tf.gather_nd(viewRec, index)
                self.loss_supp += tf.compat.v1.keras.losses.MSE(labels_sub, viewRec_sub)

            self.loss_main = tf.compat.v1.keras.losses.MSE(labels_sub, main_sub)

            self.loss_preds = tf.compat.v1.keras.losses.MSE(labels_sub, preds_sub)

            self.cost = Alpha * self.loss_supp + \
                        Beta * self.loss_preds + \
                        1 * self.loss_main

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

            self.opt_op = self.optimizer.minimize(self.cost)
            self.grads_vars = self.optimizer.compute_gradients(self.cost)


class SLMGAE():
    def __init__(self, placeholders, num_features, features_nonzero, num_nodes, num_supView, name, hid1,hid2,Coe):
        self.name = name
        self.num_nodes = num_nodes
        self.num_supView = num_supView
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adjs = placeholders['support']
        self.dropout = placeholders['dropout']
        self.inputs = placeholders['features']
        self.hid1=hid1
        self.hid2=hid2
        self.Coe=Coe
        self.support_recs = []

        with tf.compat.v1.variable_scope(self.name):
            self.attentionLayer = AttentionRec(
                name='Attention_Layer',
                output_dim=self.num_nodes,
                num_support=self.num_supView,
                act=lambda x: x)

            self.build()

    def build(self):
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer1',
            input_dim=self.input_dim,
            output_dim=self.hid1,
            adj=self.adjs[0],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden2 = GraphConvolutionSparse(
            name='gcn_sparse_layer2',
            input_dim=self.input_dim,
            output_dim=self.hid1,
            adj=self.adjs[1],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden3 = GraphConvolutionSparse(
            name='gcn_sparse_layer3',
            input_dim=self.input_dim,
            output_dim=self.hid1,
            adj=self.adjs[2],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden4 = GraphConvolutionSparse(
            name='gcn_sparse_layer3',
            input_dim=self.input_dim,
            output_dim=self.hid1,
            adj=self.adjs[3],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden5 = GraphConvolution(
            name='gcn_dense_layer1',
            input_dim=self.hid1,
            output_dim=self.hid2,
            adj=self.adjs[0],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden1)

        self.hidden6 = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.hid1,
            output_dim=self.hid2,
            adj=self.adjs[1],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden2)

        self.hidden7 = GraphConvolution(
            name='gcn_dense_layer3',
            input_dim=self.hid1,
            output_dim=self.hid2,
            adj=self.adjs[2],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden3)

        self.hidden8 = GraphConvolution(
            name='gcn_dense_layer3',
            input_dim=self.hid1,
            output_dim=self.hid2,
            adj=self.adjs[3],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden4)

        self.support_recs.append(InnerProductDecoder(
                name='gcn_decoder1',
                output_dim=self.hid2,
                act=lambda x: x)(self.hidden5))

        self.support_recs.append(InnerProductDecoder(
                name='gcn_decoder2',
                output_dim=self.hid2,
                act=lambda x: x)(self.hidden6))

        self.support_recs.append(InnerProductDecoder(
                name='gcn_decoder3',
                output_dim=self.hid2,
                act=lambda x: x)(self.hidden7))

        # self.att = tf.reduce_mean(self.support_recs)
        self.att = self.attentionLayer(self.support_recs)

        self.main_rec = InnerProductDecoder(
                name='gcn_decoder_main',
                output_dim=self.hid2,
                act=lambda x: x)(self.hidden8)

        self.reconstructions = tf.add(self.main_rec, tf.multiply(self.Coe, self.att))

    def predict(self):
        return self.reconstructions
