## @package obstacke-detection
#  Neural network
#
#  This module can be used to load and run the trained networks
#  It can be used by importing the package, or this module in a python module.

import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import models.inception

class Network():

    def __init__(self):
        tf.reset_default_graph()
        config = tf.ConfigProto(
        device_count = {'GPU': 0}
        )
        self._sess = tf.Session(config = config)
        saver = tf.train.import_meta_graph(os.path.join(os.path.dirname(os.path.abspath('__file__')),'models','stage3_final','model_final.meta'))
        saver.restore(self._sess,tf.train.latest_checkpoint(os.path.join(os.path.dirname(os.path.abspath('__file__')),'models','stage3_final')))
        print('Model restored')    
        graph = tf.get_default_graph() 

        iterator_init = graph.get_operation_by_name('iterator_init_op')
        valid_init = graph.get_operation_by_name('valid_init_op')
        train_step = graph.get_operation_by_name('Fine_tuning/train_op')
        self._mask = graph.get_tensor_by_name('Fine_tuning/Th_prediction:0')
        self._pred = graph.get_tensor_by_name('Fine_tuning/Prediction:0')
        loss = graph.get_tensor_by_name('Fine_tuning/Loss:0')
        labels = graph.get_tensor_by_name('Labels:0')    
        loss_weight = graph.get_tensor_by_name('Fine_tuning/Loss_weight:0')
        self._inputs = graph.get_tensor_by_name('Inputs:0')
        avg_valid = graph.get_tensor_by_name('avg_valid:0')

    def predict(self, data):
        p,m = self._sess.run([self._pred,self._mask], feed_dict={self._inputs: [data]})
        return (p,m)

    def __del__(self):
        self._sess.close()
