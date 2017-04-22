import numpy as np
import tensorflow as tf

def layer_wise_building_block(hidden_activation, hidden_units):
    input_w = tf.get_variable("input_Weight", [784, hidden_units], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    input_b = tf.Variable(tf.zeros([hidden_units]))
    return tf.matmul(hidden_activation, input_w) + input_b
