import os
import tensorflow as tf
from batch_generator import TrainValBatchGenerator
from data_handler import DataHandler
import numpy as np
import pandas as pd
from collections import defaultdict


class Embedder():
    def __init__(self, *, is_training, inputs, target_dict, corrupt_prop_matrix,
                 num_of_vars,
                 total_num_of_values,
                 num_of_values_by_var,
                 embedding_size):

        self.is_training = is_training
        self.embedding_size = embedding_size

        with tf.variable_scope('embedding_layer'):
            self.variable_embeddings = tf.get_variable(name='variable_embeddings',
                                                       shape=[total_num_of_values,
                                                              embedding_size],
                                                       initializer=tf.truncated_normal_initializer())

        embed_variable_all =tf.nn.embedding_lookup(self.variable_embeddings,
                                                   np.arange(total_num_of_values))

        embed_variable_corrupt = tf.matmul(corrupt_prop_matrix, embed_variable_all)
        #TODO confusing becuase value computation is done by matrix multiplication
        #TODO embed_variable embedding is done by lookup
        embed_variable = tf.nn.embedding_lookup(self.variable_embeddings, inputs)
        embedding = tf.add(tf.reduce_sum(embed_variable, reduction_indices=1),
                           embed_variable_corrupt)

        self.embedding = embedding
        self.optimizer = dict()
        self.predictions = dict()
        self.loss = dict()
        self.accuracy = dict()
        self.recall = dict()
        self.precision = dict()
        self.f1 = dict()
        self.summary = defaultdict(lambda :list())
        self.summary_update_op =defaultdict(lambda :list())

        for key, num_of_values_in_var in enumerate(num_of_values_by_var):
            self.add_output_layer(target=target_dict[key], var_idx=key,
                                      num_of_values_in_var=num_of_values_in_var)
        # self.add_summary(num_of_values_by_var, target_dict) #TODO is this right?
            # should the summary operiation be separated from the model code?

    def add_output_layer(self, target, var_idx, num_of_values_in_var):
        with tf.variable_scope(str(var_idx)):
            with tf.variable_scope('output_layer'):
                softmax_weights = tf.get_variable(name='softmax_weights',
                                                  shape=[self.embedding_size,
                                                         num_of_values_in_var],
                                                  initializer=tf.truncated_normal_initializer())
                softmax_biases = tf.get_variable(name='softmax_biases',
                                                 shape=[num_of_values_in_var],
                                                 initializer=tf.zeros_initializer())
                logits = tf.matmul(self.embedding, softmax_weights) + softmax_biases
            with tf.variable_scope('loss'):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,labels=target,name='loss'))
                self.loss[var_idx] = loss

            if self.is_training:
                with tf.variable_scope('train'):
                    self.optimizer[var_idx] = tf.train.AdamOptimizer().minimize(loss)
            with tf.variable_scope('prediction'):
                self.predictions[var_idx] = tf.cast(tf.argmax(logits, 1), tf.int32)
            with tf.variable_scope('accuracy'):
                self.accuracy[var_idx] = tf.reduce_mean(
                    tf.cast(tf.equal(self.predictions[var_idx], target), tf.float32),
                    name='accuracy')

            with tf.variable_scope('metrics'):
                for var_value in range(num_of_values_in_var):
                    target_of_var_value = tf.equal(target, var_value)
                    pred_of_var_value = tf.equal(self.predictions[var_idx], var_value)
                    recall, update_recall = tf.contrib.metrics.streaming_recall(
                        pred_of_var_value,
                                                                  target_of_var_value)
                    precision, update_precision = tf.contrib.metrics.streaming_precision(
                        pred_of_var_value,
                                                                     target_of_var_value)
                    f1 = 2 * precision * recall / (recall +
                                                   precision)

                    self.summary_update_op[(var_idx, var_value)].append(update_recall)
                    self.summary_update_op[(var_idx, var_value)].append(update_precision)
                    self.recall[(var_idx, var_value)] = recall
                    self.precision[(var_idx, var_value)] = precision
                    self.f1[(var_idx, var_value)] = f1


# Train Embedding & Prepare for Visualization
def train_embedder(data, configuration):

    train_batch_size=configuration.train_batch_size
    val_batch_size=configuration.val_batch_size

    embedding_size=configuration.embedding_size
    max_iteration = configuration.max_iteration

    print_loss_every = configuration.print_loss_every

    LOG_DIR = configuration.LOG_DIR
    model_save_filename = configuration.model_save_filename
    metadata_filename = configuration.metadata_filename
    corruption_ratio = configuration.corruption_ratio

    data_handler = DataHandler(data)  # Pandas format data
    num_of_vars = data_handler.num_of_vars
    batch_generator = TrainValBatchGenerator(
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size, data_handler=data_handler,
        corruption_ratio=corruption_ratio
        )

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True


    with tf.Session(config=sess_config) as sess:
        input_shape = num_of_vars - batch_generator.num_corruption - 1 #target
        inputs = tf.placeholder(tf.int32, shape=[None,input_shape],name='input')

        corrupt_prop_matrix = tf.placeholder(tf.float32, shape=[None, data_handler.total_num_of_values],
                                        name='corrupt_input')

        target_dict = dict()
        for key in range(num_of_vars):
            with tf.variable_scope('target_{}'.format(key)):
                target_dict[key] = tf.placeholder(tf.int32, shape=[None],
                                                  name='target_{}'.format(key))
        with tf.variable_scope('Embedder'):
            embedder_train = Embedder(is_training=True,
                                      inputs=inputs,
                                      num_of_vars=num_of_vars,
                                      total_num_of_values=data_handler.total_num_of_values,
                                      num_of_values_by_var=data_handler.num_of_values_by_var,
                                      target_dict=target_dict,
                                      corrupt_prop_matrix=corrupt_prop_matrix,
                                      embedding_size=embedding_size
                                      )
            for key, num_of_values_in_var in enumerate(data_handler.num_of_values_by_var):
                embedder_train.summary[key].append(tf.summary.scalar('Train_loss_{}'.format(
                    key),embedder_train.loss[key]))
                embedder_train.summary[key].append(tf.summary.scalar('Train_accuracy_{}'.format(
                    key), embedder_train.accuracy[key]))
                for var_value in range(num_of_values_in_var):
                    embedder_train.summary[key].append(
                        tf.summary.scalar('Train_recall_{}_{}'.format(key,var_value),
                                          embedder_train.recall[(key, var_value)]))
                    embedder_train.summary[key].append(
                        tf.summary.scalar('Train_precision_{}_{}'.format(key,var_value),
                                          embedder_train.precision[(key, var_value)]))
                    embedder_train.summary[key].append(tf.summary.scalar('Train_f1_{}_{}'.format(
                        key, var_value),embedder_train.f1[(key, var_value)]))


        with tf.variable_scope('Embedder', reuse=True):
            embedder_val = Embedder(is_training=False,
                                    inputs=inputs,
                                    num_of_vars=num_of_vars,
                                    total_num_of_values=data_handler.total_num_of_values,
                                    num_of_values_by_var=data_handler.num_of_values_by_var,
                                    target_dict=target_dict,
                                    corrupt_prop_matrix=corrupt_prop_matrix,
                                    embedding_size=embedding_size,
                                    )
            for key, num_of_values_in_var in enumerate(data_handler.num_of_values_by_var):
                embedder_val.summary[key].append(tf.summary.scalar('Validation_loss_{}'.format(
                    key),embedder_val.loss[key]))
                embedder_val.summary[key].append(tf.summary.scalar('Validation_accuracy_{}'.format(
                    key), embedder_val.accuracy[key]))
                for var_value in range(num_of_values_in_var):
                    embedder_val.summary[key].append(
                        tf.summary.scalar('Validation_recall_{}_{}'.format(key,var_value),
                                          embedder_val.recall[(key, var_value)]))
                    embedder_val.summary[key].append(
                        tf.summary.scalar('Validation_precision_{}_{}'.format(key,var_value),
                                          embedder_val.precision[(key, var_value)]))
                    embedder_val.summary[key].append(tf.summary.scalar('Validation_f1_{}_{}'.format(
                        key, var_value),embedder_val.f1[(key, var_value)]))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        summary_writer = tf.summary.FileWriter(LOG_DIR)
        summary_writer.add_graph(tf.get_default_graph())

        def eval(fetch_dict, batch, summary=False):
            keys = list(fetch_dict.keys())
            for var_idx, vars_in_batch in enumerate(batch):
                if type(keys[0]) is tuple:
                    fetch_value = [fetch_dict[x] for x in keys if x[0] == var_idx]
                else:
                    fetch_value = fetch_dict[var_idx]
                inputs_, corrupt_prop_matrix_, target_ = vars_in_batch
                fetch_value = sess.run(fetch_value,
                                       feed_dict={inputs:inputs_,
                                                  target_dict[var_idx]:target_,
                                                  corrupt_prop_matrix:corrupt_prop_matrix_})
                if summary:
                    for val in fetch_value:
                        summary_writer.add_summary(val, i+1)

        for i in range(max_iteration):
            print('Epoch:', i)
            batch_train = batch_generator.next_train_batch()

            eval(embedder_train.optimizer, batch_train)

            if (i + 1) % print_loss_every == 0:
                eval(embedder_train.summary_update_op, batch_train)
                eval(embedder_train.summary, batch_train, True)

                batch_val = batch_generator.next_val_batch()
                eval(embedder_val.summary_update_op, batch_val)
                eval(embedder_val.summary, batch_val, True)

                summary_writer.flush()

        saver.save(sess, save_path=os.path.join(LOG_DIR,model_save_filename),
                            global_step=(i + 1))

        summary_writer.close()