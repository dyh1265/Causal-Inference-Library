from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout
import tensorflow_probability as tfp


tfpl = tfp.layers
tfd = tfp.distributions

class FullyConnected(Layer):
    def __init__(self, n_fc, hidden_phi, out_size,  final_activation, name, kernel_reg, kernel_init, activation='elu',
                 bias_initializer=None, dropout=False, batch_norm=False, use_bias=True,  dropout_rate=0.0, **kwargs):
        super(FullyConnected, self).__init__(name=name, **kwargs)
        self.Layers = []
        for i in range(n_fc-1):
            if batch_norm:
                self.Layers.append(BatchNormalization())
            self.Layers.append(Dense(units=hidden_phi, activation=activation, kernel_initializer=kernel_init,
                                     bias_initializer=bias_initializer, use_bias=use_bias,
                                     kernel_regularizer=kernel_reg, name=name + str(i)))
            if dropout:
                self.Layers.append(Dropout(dropout_rate))
        self.Layers.append(Dense(units=out_size, activation=final_activation, name=name + 'out'))

    def call(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x
