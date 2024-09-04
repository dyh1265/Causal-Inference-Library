import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from utils.layers import FullyConnected, VariationalFullyConnected, Convolutional1D, LocallyConnected
# from models.CausalModel import CausalModel
from models.CausalModel import *
import keras_tuner as kt
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping
import os, sys
import tensorflow_probability as tfp
tf.get_logger().setLevel(logging.ERROR)
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
import json
from os.path import exists

os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt

plt.show()
import json
from os.path import exists
import shutil


def callbacks(rlr_monitor):
    cbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(monitor=rlr_monitor, factor=0.5, patience=5, verbose=0, mode='auto',
                          min_delta=0., cooldown=0, min_lr=1e-8),
        EarlyStopping(monitor='val_regression_loss', patience=40, min_delta=0., restore_best_weights=False)
    ]
    return cbacks


class HyperGNNTarnet(kt.HyperModel, CausalModel):
    def __init__(self, params, name='gnn_tarnet'):
        super().__init__()
        self.params = params
        self.name = name

    def build(self, hp):
        momentum = 0.9

        model = GNNTARnetModel(
            name=self.name,
            hp=hp,
            params=self.params,
        )

        model.compile(optimizer=SGD(learning_rate=self.params['lr'], nesterov=True, momentum=momentum),
                      loss=self.regression_loss,
                      metrics=self.regression_loss, run_eagerly=False
                      )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class GraphConvLayer(layers.Layer):
    def __init__(
            self,
            params,
            gnn_n_fc,
            gnn_hidden_units,
            *args,
            **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.params = params
        self.aggregation_type = params['aggregation_type']
        self.combination_type = params['combination_type']
        self.normalize = params['normalize']
        self.gnn_n_fc = gnn_n_fc
        self.gnn_hidden_units = gnn_hidden_units

        self.ffn_prepare = FullyConnected(n_fc=self.gnn_n_fc, hidden_phi=self.gnn_hidden_units,
                                          final_activation='elu', out_size=self.gnn_hidden_units,
                                          kernel_init=self.params['kernel_init'], use_bias=False,
                                          kernel_reg=None, dropout=False, dropout_rate=self.params['dropout_rate'],
                                          name='ffn_prepare')

        self.update_fn = FullyConnected(n_fc=self.gnn_n_fc, hidden_phi=self.gnn_hidden_units,
                                        final_activation=None, out_size=self.gnn_hidden_units, use_bias=True,
                                        kernel_init=self.params['kernel_init'], batch_norm=False,
                                        kernel_reg=None, dropout=False, dropout_rate=self.params['dropout_rate'],
                                        name='update_fn')

    def prepare(self, node_representations):
        # node_representations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_representations)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_representations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_nodes = node_representations.shape[1]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(tf.transpose(neighbour_messages, [1, 0, 2]),
                                                              node_indices, num_segments=num_nodes)
            aggregated_message = tf.transpose(aggregated_message, [1, 0, 2])
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(tf.transpose(neighbour_messages, [1, 0, 2]),
                                                               node_indices, num_segments=num_nodes)
            aggregated_message = tf.transpose(aggregated_message, [1, 0, 2])
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(neighbour_messages,
                                                              node_indices, num_segments=num_nodes)
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")
        return aggregated_message

    def update(self, node_representations, aggregated_messages):
        # node_representations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "concat":
            # Concatenate the node_representations and aggregated_messages.
            h = tf.concat([node_representations, aggregated_messages], axis=2)
        elif self.combination_type == "add":
            # Add node_representations and aggregated_messages.
            h = node_representations + aggregated_messages
        elif self.combination_type == "mlp":
            h = node_representations * aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.update_fn(h)
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_representations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """
        node_representations, edges, edge_weights = inputs
        # Get node_indices (source) and parent_indices (target) from edges.
        parent_indices, node_indices = edges[:, 0], edges[:, 1]
        parents_repesentations = tf.gather(node_representations, parent_indices, axis=1)
        # Prepare the messages of the parents.
        parent_messages = self.prepare(parents_repesentations)
        # Aggregate the parents messages.
        aggregated_messages = self.aggregate(node_indices, parent_messages, node_representations)
        return self.update(node_representations, aggregated_messages)


class Embedding(Model):
    def __init__(self, params, vector_size, num_neurons):
        super(Embedding, self).__init__()
        self.vector_size = vector_size
        self.num_neurons = num_neurons
        self.params = params
        self.networks = []
        for i in range(vector_size):
            x = FullyConnected(n_fc=1, hidden_phi=1,
                                      final_activation=None, out_size=self.num_neurons,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y'+str(i))
            self.networks.append(x)

    def call(self, inputs):
        outputs = []
        for i, layer in enumerate(self.networks):
            x_i = inputs[:, i]
            x = layer(x_i)
            outputs.append(x)
        outputs = tf.stack(outputs, axis=1)
        return outputs

class GNNTARnetModel(Model):
    def __init__(
            self,
            name,
            params,
            hp,
            *args,
            **kwargs,
    ):
        super(GNNTARnetModel, self).__init__(*args, **kwargs)
        self.params = params
        self.model_name = name
        self.edges = params['edges']
        self.gnn_weights = params['weights']
        # self.gnn_n_fc = self.params['gnn_n_fc']
        # self.gnn_hidden_units = self.params['gnn_hidden_units']
        self.gnn_n_fc = hp.Int('gnn_n_fc', min_value=2, max_value=10, step=1)
        self.gnn_hidden_units = hp.Int('gnn_hidden_units', min_value=16, max_value=256, step=16)
        self.n_hidden_0 = hp.Int('n_hidden_0', min_value=2, max_value=10, step=1)
        self.hidden_y0 = hp.Int('hidden_y0', min_value=16, max_value=256, step=16)
        self.n_hidden_1 = hp.Int('n_hidden_1', min_value=2, max_value=10, step=1)
        self.hidden_y1 = hp.Int('hidden_y1', min_value=16, max_value=256, step=16)

        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            params=self.params,
            gnn_n_fc=self.gnn_n_fc,
            gnn_hidden_units=self.gnn_hidden_units,
            name="graph_conv1"
        )

        # # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            params=self.params,
            gnn_n_fc=self.gnn_n_fc,
            gnn_hidden_units=self.gnn_hidden_units,
            name="graph_conv2"
        )

        self.embedding = Embedding(params=self.params, vector_size=self.params['num_nodes'], num_neurons=self.gnn_hidden_units)

        self.pred_y0 = FullyConnected(n_fc=self.n_hidden_0, hidden_phi=self.hidden_y0,
                                      final_activation=self.params['activation'], out_size=1,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y0')

        self.pred_y1 = FullyConnected(n_fc=self.n_hidden_1, hidden_phi=self.hidden_y1,
                                      final_activation=self.params['activation'], out_size=1,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y1')


        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = inputs
        x = self.embedding(x)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, None))
        # # Skip connection.
        x = x1 + x
        # # # # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, None))
        x = x2 + x
        # # # use info about nodes influencing the outcome
        x = tf.gather(x, self.params['influence_y'], axis=1)
        # # Flatten
        x = self.flatten(x)
        # Make a prediction
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        # Concatenate the result and return
        concat_pred = tf.concat([y0_pred, y1_pred], axis=-1)
        return concat_pred


class GNNTARnetHyper(CausalModel):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name = None

    def fit_tuner(self, seed=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        edges = kwargs['edges']
        weights = kwargs['weights']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)

        directory_name = 'params_' + self.params['tuner_name'] + '/' + self.params['dataset_name']
        setSeed(seed)

        project_name = self.params["model_name"]

        self.params['edges'] = edges
        self.params['weights'] = weights
        self.params['num_edges'] = edges.shape[0]
        self.params['num_nodes'] = x.shape[1]

        hp = kt.HyperParameters()

        self.directory_name = directory_name
        self.project_name = project_name

        hypermodel = HyperGNNTarnet(params=self.params, name='gnn_tarnet_search')
        objective = kt.Objective("val_regression_loss", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='regression_loss', patience=5)]
        tuner.search(x, yt, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=self.params['verbose'])

        return

    def fit_model(self, seed=0, count=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        edges = kwargs['edges']
        weights = kwargs['weights']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)
        setSeed(seed)

        self.params['edges'] = edges
        self.params['weights'] = weights
        self.params['num_edges'] = edges.shape[0]

        tuner = self.params['tuner'](
            HyperGNNTarnet(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {'gnn_n_fc': self.params['gnn_n_fc'],
                                'gnn_hidden_units': self.params['gnn_hidden_units'],
                                'n_hidden_0': self.params['n_hidden_0'],
                               'hidden_y0': self.params['hidden_y0'],
                               'n_hidden_1': self.params['n_hidden_1'],
                               'hidden_y1': self.params['hidden_y1']}
        else:
            self.params['gnn_n_fc'] = best_hps.get('gnn_n_fc')
            self.params['gnn_hidden_units'] = best_hps.get('gnn_hidden_units')
            self.params['n_hidden_0'] = best_hps.get('n_hidden_0')
            self.params['hidden_y0'] = best_hps.get('hidden_y0')
            self.params['n_hidden_1'] = best_hps.get('n_hidden_1')
            self.params['hidden_y1'] = best_hps.get('hidden_y1')

        model = tuner.hypermodel.build(best_hps)
        stop_early = [
            ReduceLROnPlateau(monitor='regression_loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=0., cooldown=0, min_lr=1e-8),
            EarlyStopping(monitor='regression_loss', patience=40, restore_best_weights=True)]

        model.fit(x=x, y=yt,
                  validation_split=0.0,
                  callbacks=stop_early,
                  epochs=self.params['epochs'],
                  verbose=self.params['verbose'],
                  batch_size=self.params['batch_size'])

        if count == 0:
            print(model.summary())
            self.sparams = f""" gnn_n_fc = {best_hps.get('gnn_n_fc')} gnn_hidden_units = {best_hps.get('gnn_hidden_units')}
             n_hidden_0 = {best_hps.get('n_hidden_0')} n_hidden_1 = {best_hps.get('n_hidden_1')}
             hidden_y0 = {best_hps.get('hidden_y0')}  hidden_y1 = {best_hps.get('hidden_y1')}"""
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                              {self.sparams}""")
        return model

    def load_graph(self, path):
        if self.params['json']:
            with open(path) as f:
                graph = json.load(f)
        else:
            graph = pd.read_csv(path, header=None)
        return graph

    def get_graph_info(self, graph):
        if self.params['json']:
            edges = np.concatenate([np.asarray(graph['from']).reshape(-1, 1),
                                    np.asarray(graph['to']).reshape(-1, 1)], axis=1)
            influence_y = np.asarray(graph['influence_y'])
            edge_weights = np.asarray(graph['weights'])
            edge_weights = np.expand_dims(edge_weights / np.sum(edge_weights, axis=0), axis=-1)
            graph_info = {'edges': edges, 'edge_weights': edge_weights, 'influence_y': influence_y}
            return graph_info
        else:
            edges = np.asarray(graph)
            influence_y = []
            """Get non-zero elements from acyclic_W for edges and stack them to match the num of patients.
            Create an edges array (sparse adjacency matrix) of shape [num_samples, 2, num_edges]."""

            """Create an edge weights array of ones."""
            edge_weights = np.ones(shape=(edges.shape[0]))
            edge_weights = np.expand_dims(edge_weights / np.sum(edge_weights, axis=0), axis=-1)

            graph_info = {'edges': edges, 'edge_weights': edge_weights, 'influence_y': influence_y}
            return graph_info
    def load_graphs(self, x_train, count):

        if self.dataset_name == 'sum':
                path = 'graphs/sum_graph_' + str(self.params['num_layers'])
        else:
            path = 'graphs/' + self.params['dataset_name'] + '_' + self.params['cdm']

        if not self.params['json']:
            file_name = '/graph_' + str(count) + '.csv'
        else:
            file_name = '/graph_' + str(count) + '.json'

        graph = self.load_graph(path + file_name)
        graph_info = self.get_graph_info(graph)

        if self.params['eye']:
            acyclic_W = np.eye(x_train.shape[1])
            graph = np.asarray(np.nonzero(acyclic_W))
            edges = np.transpose(graph)
            graph_info['influence_y'] = np.arange(x_train.shape[1])
            graph_info['edges'] = edges
        return graph_info

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)

    def prepare_data(self, **kwargs):
        data_train, data_test = self.load_data(**kwargs)
        x_train, t_train = data_train['x'], data_train['t']
        self.params['num_nodes'] = x_train.shape[1]
        self.folder_ind = kwargs.get('folder_ind')
        x_test = np.expand_dims(data_test['x'], axis=-1)
        x_train = np.expand_dims(x_train, axis=-1)

        graph_info = self.load_graphs(x_train=x_train, count=kwargs.get('count'))

        influence_y = graph_info['influence_y']
        self.params['num_edges'] = len(graph_info['edges'][1])
        self.params['influence_y'] = np.append(influence_y, np.expand_dims(x_train.shape[1], axis=0))

        edges = graph_info['edges']
        weights = graph_info['edge_weights']
        if self.params['binary']:
            y_train = data_train['y']
        else:
            y_train = data_train['ys']

        # x_train[:, :] = 0
        # x_test[:, :] = 0

        args_train = {'x': x_train, 't': t_train, 'y': y_train, 'edges': edges, 'weights': weights}

        return x_train, x_test, args_train, data_train, data_test

    def train_and_evaluate(self, metric_list_train, metric_list_test, average_metric_list_train,
                           average_metric_list_test, **kwargs):
        setSeed(seed=42)
        tracker_test, tracker_train = self.get_trackers(count=kwargs.get('count'))
        x_train, x_test, args_train, data_train, data_test = self.prepare_data(**kwargs)

        with tracker_train:
            self.fit_tuner(seed=0, **args_train)
            model = self.fit_model(**args_train, count=kwargs.get('count'))
        self.emission_train.append(tracker_train.final_emissions)

        # make a prediction
        with tracker_test:
            concat_pred_test = self.evaluate(x_test, model)
            concat_pred_train = self.evaluate(x_train, model)
        self.emission_test.append(tracker_test.final_emissions)

        y0_pred_test, y1_pred_test = concat_pred_test[:, 0], concat_pred_test[:, 1]
        y0_pred_test = tf.expand_dims(y0_pred_test, axis=1)
        y1_pred_test = tf.expand_dims(y1_pred_test, axis=1)

        y0_pred_train, y1_pred_train = concat_pred_train[:, 0], concat_pred_train[:, 1]
        y0_pred_train = tf.expand_dims(y0_pred_train, axis=1)
        y1_pred_train = tf.expand_dims(y1_pred_train, axis=1)

        if self.params['dataset_name'] == 'jobs':
            _, policy_risk_test, _, test_ATT = self.find_policy_risk(y0_pred_test, y1_pred_test, data_test)
            _, policy_risk_train, _, train_ATT = self.find_policy_risk(y0_pred_train, y1_pred_train, data_train)

            print(kwargs.get('count'), 'Policy Risk Test = ', policy_risk_test, '| Test ATT', test_ATT,
                  '| Policy Risk Train = ', policy_risk_train, '| Train ATT', train_ATT)
            metric_list_test.append(policy_risk_test)
            metric_list_train.append(policy_risk_train)

            average_metric_list_test.append(test_ATT)
            average_metric_list_train.append(train_ATT)

        else:
            pehe_test, ate_test = self.find_pehe(y0_pred_test, y1_pred_test, data_test)
            pehe_train, ate_train = self.find_pehe(y0_pred_train, y1_pred_train, data_train)

            if self.params['dataset_name'] == 'acic' or self.params['dataset_name'] == 'gnn':
                print(kwargs.get('folder_ind'), kwargs.get('count'), 'Pehe Test = ', pehe_test, 'Pehe Train = ',
                      pehe_train)
            else:
                print(kwargs.get('count'), 'Pehe Test = ', pehe_test, ' Pehe Train = ', pehe_train, ' ATE test = ',
                      ate_test,
                      ' ATE train = ', ate_train)

            metric_list_test.append(pehe_test)
            metric_list_train.append(pehe_train)

            average_metric_list_test.append(ate_test)
            average_metric_list_train.append(ate_train)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

