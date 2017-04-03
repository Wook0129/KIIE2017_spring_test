import config
import os
import tensorflow as tf
from batch_generator import TrainValBatchGenerator
from data_handler import DataHandler
import numpy as np


class Embedder:
    def __init__(self, *, inputs, corrupt_prop_matrix,
                 num_of_vars,
                 total_num_of_values,
                 embedding_size):

        self.embedding_size = embedding_size
        with tf.variable_scope('embedding_layer'):
            self.variable_embeddings = tf.get_variable(name='variable_embeddings',
                                                       shape=[total_num_of_values,
                                                              embedding_size],
                                                       initializer=tf.random_normal_initializer())

        embed_variable_all =tf.nn.embedding_lookup(self.variable_embeddings,
                                                   np.arange(total_num_of_values))

        embed_variable_corrupt = tf.matmul(corrupt_prop_matrix, embed_variable_all)
        #TODO confusing becuase value computation is done by matrix multiplication
        #TODO embed_variable embedding is done by lookup
        embed_variable = tf.nn.embedding_lookup(self.variable_embeddings, inputs)
        embedding = tf.add(tf.reduce_sum(embed_variable, reduction_indices=1),
                           embed_variable_corrupt)

        self.embedding = embedding
        self.loss = dict()
        self.optimizer = dict()
        self.predictions = dict()
        self.accuracy = dict()

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
            with tf.variable_scope('train'):
                self.optimizer[var_idx] = tf.train.AdamOptimizer(
                    config.learning_rate).minimize(loss)
            with tf.variable_scope('prediction'):
                self.predictions[var_idx] = tf.equal(tf.cast(tf.argmax(logits, 1),
                                                             tf.int32), target)
            with tf.variable_scope('evaluation'):
                self.accuracy[var_idx] = tf.reduce_mean(
                    tf.cast(self.predictions[var_idx], "float"))


# Train Embedding & Prepare for Visualization
def train_embedder(data, *, max_iteration=config.max_iteration,
                   train_batch_size=config.train_batch_size,
                   val_batch_size=config.val_batch_size,
                   embedding_size=config.embedding_size):
    data_handler = DataHandler(data)  # Pandas format data
    num_of_vars = data_handler.num_of_vars
    batch_generator = TrainValBatchGenerator(
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size, data_handler=data_handler,
        corruption_ratio=0.2
        )

    with tf.Session() as sess:
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
            embedder = Embedder(inputs=inputs,
                                num_of_vars=num_of_vars,
                                total_num_of_values=data_handler.total_num_of_values,
                                corrupt_prop_matrix=corrupt_prop_matrix,
                                embedding_size=embedding_size)
            for key, num_of_values_in_var in enumerate(data_handler.num_of_values_by_var):
                embedder.add_output_layer(target=target_dict[key], var_idx=key,
                                          num_of_values_in_var=num_of_values_in_var)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(config.LOG_DIR)
        summary_writer.add_graph(tf.get_default_graph())
        summary_writer.close()

        def eval(fetch, batch, data_type, print_fetch):
            for key, vars_in_batch in enumerate(batch):
                inputs_, corrupt_prop_matrix_, target_ = vars_in_batch
                fetch_value = sess.run(fetch[key], feed_dict={inputs:inputs_,
                                                             target_dict[key]:target_,
                                                              corrupt_prop_matrix:
                                                                  corrupt_prop_matrix_})
                if print_fetch:
                    print('{} {} at step {} key {} : {}'.format(data_type, print_fetch,
                                                                i + 1,key,fetch_value))

        for i in range(max_iteration):
            batch_train = batch_generator.next_train_batch()

            eval(embedder.optimizer, batch_train, None, None)

            if (i + 1) % config.print_loss_every == 0:
                eval(embedder.loss, batch_train, 'Train', 'Loss')
                eval(embedder.accuracy, batch_train, 'Train', 'Accuracy')

                batch_val = batch_generator.next_val_batch()

                eval(embedder.loss, batch_val, 'Validation', 'Loss')
                eval(embedder.accuracy, batch_val, 'Validation', 'Accuracy')

        saver.save(sess, save_path=os.path.join(config.LOG_DIR,config.model_save_filename),
                            global_step=(i + 1))

        # Save And Visualize Embeddings
        tensorboard_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        var_embedding = tensorboard_config.embeddings.add()
        var_embedding.tensor_name = embedder.variable_embeddings.name
        var_embedding.metadata_path = os.path.join(config.LOG_DIR,
                                                   config.metadata_filename)

        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer,
                                                                      tensorboard_config)
