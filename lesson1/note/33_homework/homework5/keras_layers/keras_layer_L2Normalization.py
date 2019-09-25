'''
定制的 Keras 层用于 L2-normalization.
'''

from __future__ import division
import numpy as np
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

class L2Normalization(Layer):
    '''
    用于 L2 normalization 的层, 理论参见论文 "Parsenet: Looking Wider to See Better".

    Arguments:
        gamma_init (int): 初始化的参数, 默认值为20.

    Input shape:
        4 维tensor, 如果 `dim_ordering = 'th'`,  形状为 `(batch, channels, height, width)`
                    如果 `dim_ordering = 'tf'`,  形状为 `(batch, height, width, channels)`

    Returns:
        和输入一样形状的tensor. 

    References:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    '''

    def __init__(self, gamma_init=20, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        return output * self.gamma

    def get_config(self):
        config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
