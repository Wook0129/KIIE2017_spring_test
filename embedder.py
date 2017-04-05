import os
import tensorflow as tf
from batch_generator import TrainValBatchGenerator
from data_handler import DataHandler
import numpy as np
import pandas as pd
from collections import defaultdict


class Embedder:
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
        self.summary_op =defaultdict(lambda :list())

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

                    self.summary_op[(var_idx, var_value)].append(update_recall)
                    self.summary_op[(var_idx, var_value)].append(update_precision)
                    self.recall[(var_idx, var_value)] = recall
                    self.precision[(var_idx, var_value)] = precision
                    self.f1[(var_idx, var_value)] = f1

    def add_summary(self, num_of_values_by_var, target_dict):
        for key, num_of_values_in_var in enumerate(num_of_values_by_var):
            self.add_output_layer(target=target_dict[key], var_idx=key,
                                            num_of_values_in_var=num_of_values_in_var)
            self.summary[key].append(tf.summary.scalar('loss_{}'.format(
                key), self.loss[key]))
            self.summary[key].append(tf.summary.scalar('accuracy_{}'.format(
                key), self.accuracy[key]))
            for var_value in range(num_of_values_in_var):
                self.summary[key].append(
                    tf.summary.scalar('recall_{}_{}'.format(key, var_value),
                                      self.recall))
                self.summary[key].append(
                    tf.summary.scalar('precision_{}_{}'.format(key, var_value),
                                      self.precision))
                self.summary[key].append(tf.summary.scalar('f1_{}_{}'.format(
                    key, var_value), self.f1))


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

    mean_evaluation_df = np.zeros((int(max_iteration / print_loss_every), 5))

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

        def eval(fetch, batch, data_type, print_fetch, summary_op=False):
            sum_fetch_value = 0
            for vars_in_batch in batch:
                for key in fetch.keys():
                    if type(key) is tuple:
                        target_key, _ = key
                    else:
                        target_key = key
                    inputs_, corrupt_prop_matrix_, target_ = vars_in_batch
                    fetch_value = sess.run(fetch[key], feed_dict={inputs:inputs_,
                                                                 target_dict[
                                                                     target_key]:target_,
                                                                  corrupt_prop_matrix:
                                                                      corrupt_prop_matrix_})

                    if print_fetch:
                        print('{} {} at step {} key {} : {}'.format(data_type, print_fetch,
                                                                    i + 1,key,fetch_value))
                        sum_fetch_value += fetch_value
                    if summary_op:
                        for val in fetch_value:
                            summary_writer.add_summary(val, i+1)

            if print_fetch == 'Loss':
                mean_evaluation_df[i//print_loss_every, 0] = sum_fetch_value / (key+ 1)
            elif print_fetch == 'Accuracy':
                mean_evaluation_df[i//print_loss_every, 1] = sum_fetch_value / (key+ 1)
            elif print_fetch == 'Recall':
                mean_evaluation_df[i // print_loss_every, 2] = sum_fetch_value / (target_key + 1)
            elif print_fetch == 'Precision':
                mean_evaluation_df[i // print_loss_every, 3] = sum_fetch_value / (target_key + 1)
            elif print_fetch == 'f1':
                mean_evaluation_df[i // print_loss_every, 4] = sum_fetch_value / (target_key + 1)

        for i in range(max_iteration):
            batch_train = batch_generator.next_train_batch()

            eval(embedder_train.optimizer, batch_train, None, None)

            if (i + 1) % print_loss_every == 0:
                eval(embedder_train.summary, batch_train, 'Train', None, True)
                eval(embedder_train.loss, batch_train, 'Train', 'Loss')
                eval(embedder_train.accuracy, batch_train, 'Train', 'Accuracy')
                eval(embedder_train.summary_op, batch_train, None, None)
                eval(embedder_train.recall, batch_train, 'Train', 'Recall')
                eval(embedder_train.precision, batch_train, 'Train', 'Precision')
                eval(embedder_train.f1, batch_train, 'Train', 'f1')

                batch_val = batch_generator.next_val_batch()
                eval(embedder_val.summary, batch_val, 'Validation', None, True)
                eval(embedder_val.loss, batch_val, 'Validation', 'Loss')
                eval(embedder_val.accuracy, batch_val, 'Validation', 'Accuracy')
                eval(embedder_val.summary_op, batch_val, None, None)
                eval(embedder_val.recall, batch_val, 'Validation', 'Recall')
                eval(embedder_val.precision, batch_val, 'Validation', 'Precision')
                eval(embedder_val.f1, batch_val, 'Validation', 'f1')
                summary_writer.flush()

        saver.save(sess, save_path=os.path.join(LOG_DIR,model_save_filename),
                            global_step=(i + 1))

        pd.DataFrame(mean_evaluation_df, columns=['loss', 'accuracy', 'recall',
                                                  'precision', 'f1'],
                     index=range(
            print_loss_every, max_iteration+1, print_loss_every),
        ).to_csv(os.path.join(LOG_DIR, 'loss_accuracy.csv'))
        summary_writer.close()