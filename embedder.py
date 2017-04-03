import config
import os
import tensorflow as tf
from batch_generator import TrainValBatchGenerator
from data_handler import DataHandler
import numpy as np


class Embedder:
    def __init__(self, *, inputs, corrupt_inputs, num_instances,
                 num_of_vars,
                 total_num_of_bins,
                 embedding_size):

        self.embedding_size = embedding_size
        with tf.variable_scope('embedding_layer'):
            self.variable_embeddings = tf.get_variable(name='variable_embeddings',
                                                       shape=[total_num_of_bins,
                                                              embedding_size],
                                                       initializer=tf.random_normal_initializer())
            self.instance_embeddings = tf.get_variable(name='instance_embeddings',
                                                       shape=[num_instances,
                                                              embedding_size],
                                                       initializer=tf.random_normal_initializer())

        # embed_variable_null = tf.nn.weighted_moments(
        #     tf.nn.embedding_lookup(self.variable_embeddings,corrupt_inputs), 1,
        #     var_prop_inputs)[0]

        # corrupt_inputs = tf.one_hot(corrupt_inputs, total_num_of_bins)

        # corrupt_inputs = tf.one_hot(pro_inputs, total_num_of_bins)

        embed_variable_all =tf.nn.embedding_lookup(self.variable_embeddings,
                                                np.arange(total_num_of_bins))

        embed_variable_null = tf.matmul(corrupt_inputs, embed_variable_all)

        embed_variable = tf.nn.embedding_lookup(self.variable_embeddings,
                                                inputs[:, :num_of_vars - 1])

        embedding = tf.add(tf.reduce_sum(embed_variable, reduction_indices=1),
                           embed_variable_null)

        self.embedding = embedding
        self.loss = dict()
        self.optimizer = dict()
        self.predictions = dict()
        self.accuracy = dict()
    def add_output_layer(self, target, variable_class, num_of_bin_in_var):
        variable_class = str(variable_class)
        with tf.variable_scope(variable_class):
            with tf.variable_scope('output_layer'):
                onehot_target = tf.squeeze(tf.one_hot(indices=target, depth=num_of_bin_in_var), 1)
                softmax_weights = tf.get_variable(name='softmax_weights',
                                                  shape=[self.embedding_size,
                                                         num_of_bin_in_var],
                                                  initializer=tf.truncated_normal_initializer())
                softmax_biases = tf.get_variable(name='softmax_biases',
                                                 shape=[num_of_bin_in_var],
                                                 initializer=tf.zeros_initializer())
                logits = tf.matmul(self.embedding, softmax_weights) + softmax_biases
            with tf.variable_scope('loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=onehot_target,
                                                            name='loss'))
                self.loss[variable_class] = loss
            with tf.variable_scope('train'):
                self.optimizer[variable_class] = tf.train.AdamOptimizer(
                    config.learning_rate).minimize(
                    loss)
            with tf.variable_scope('prediction'):
                self.predictions[variable_class] = tf.equal(tf.argmax(logits, 1),
                                            tf.argmax(onehot_target, 1))

            self.accuracy[variable_class] = tf.reduce_mean(tf.cast(self.predictions[variable_class],
                                                                   "float"))




# Train Embedding & Prepare for Visualization
def train_embedder(data, *, max_iteration=config.max_iteration,
                   train_batch_size=config.train_batch_size,
                   val_batch_size=config.val_batch_size,
                   embedding_size=config.embedding_size):
    data_handler = DataHandler(data)  # Pandas format data
    data_handler.save_metadata()

    batch_generator = TrainValBatchGenerator(
        data=data_handler.onehot_encoded_data.as_matrix(),
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size, data_handler=data_handler.get_metadata()
        )

    with tf.Session() as sess:
        inputs = tf.placeholder(tf.int32, shape=[None,data_handler.num_of_vars-2],
                                #TODO make this safe for future use
                                name='input')  # (
        # num_vars - 1) + 1 instance_id
        corrupt_var_num = 1 #TODO subject to change
        corrupt_inputs = tf.placeholder(tf.float32, shape=[None, data_handler.total_num_of_bins],
                                        name='corrupt_input')
        # var_prop_inputs = tf.placeholder(tf.float32, shape=[None, None],
        #                                  name='var_prop_input')
        target_dict = dict()

        for key, num_of_bins_in_var in enumerate(data_handler.num_of_bins_by_var): #TODO
            # 여기 key를 맞추기
            key = str(key)
            with tf.variable_scope('target_{}'.format(key)):
                target_dict[key] = tf.placeholder(tf.int32, shape=[None, 1],
                                                  name='target_{}'.format(key))

        with tf.variable_scope('Embedder'):
            embedder = Embedder(inputs=inputs,
                                num_instances=data_handler.num_instances,
                                num_of_vars=data_handler.num_of_vars,
                                total_num_of_bins=data_handler.total_num_of_bins,
                                corrupt_inputs=corrupt_inputs,
                                embedding_size=embedding_size)
            for key, num_of_bins_in_var in enumerate(data_handler.num_of_bins_by_var):
                key = str(key)
                embedder.add_output_layer(target_dict[key], key, num_of_bins_in_var)
            saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(config.LOG_DIR)
        summary_writer.add_graph(tf.get_default_graph())
        summary_writer.close()

        def eval(fetch, batch, data_type, print_fetch):

            for key, input_label in enumerate(batch):
                key = str(key)
                inputs_ = input_label[0]
                labels_ = np.asarray(input_label[1]).reshape(-1,1)
                var_prop_inputs_ = [x.pop() for x in inputs_]
                corrupt_inputs_ = [x.pop() for x in inputs_]

                tmp = np.zeros((train_batch_size, data_handler.total_num_of_bins))


                for i, idx, val in zip(range(train_batch_size), corrupt_inputs_, \
                        var_prop_inputs_):
                    tmp[i,idx] = val
                if sum([len(x) for x in inputs_]) != 60:
                    print('heelo')
                fetch_value = sess.run(fetch[key], feed_dict={inputs:inputs_,
                                                             target_dict[key]:labels_,
                                                              corrupt_inputs:
                                                                  tmp
                                                              })
                if print_fetch:
                    print('{} {} at step {} key {} : {}'.format(data_type, print_fetch,
                                                                i + 1,key,fetch_value))

        for i in range(max_iteration):
            batch = batch_generator.next_train_batch()

            # for key, train_input_label in enumerate(batch):
            #     print(key)
            #     train_key = str(key)
            #     train_inputs = train_input_label[0]
            #     train_var_prop_inputs = [x.pop() for x in train_inputs]
            #     train_corrupt_inputs = [x.pop() for x in train_inputs]
            #     train_labels = np.asarray(train_input_label[1])
            #     sess.run(embedder.optimizer[train_key],
            #              feed_dict={inputs: train_inputs, target_dict[train_key]:
            #                  train_labels, corrupt_inputs: train_corrupt_inputs,
            #                         var_prop_inputs: train_var_prop_inputs})
            eval(embedder.optimizer, batch, None, None)

            if (i + 1) % config.print_loss_every == 0:
                # for key, train_input_label in enumerate(batch):
                #     train_key = str(key)
                #     train_inputs = train_input_label[0]
                #     train_labels = train_input_label[1]
                #
                #     train_loss = sess.run(embedder.loss[train_key], feed_dict={inputs:
                #                                                            train_inputs,
                #                                                     target_dict[train_key]:
                #                                                     train_labels})
                #     print('Training Loss at step {} key {} : {}'.format(i + 1,
                #                                                         train_key,
                #                                                         train_loss))
                eval(embedder.loss, batch, 'Train', 'Loss')
                batch_val = batch_generator.next_val_batch()

                # for key, val_input_label in enumerate(batch_val):
                #     val_key = str(key)
                #     val_inputs = val_input_label[0]
                #     val_labels = val_input_label[1]
                #
                #
                #     val_loss = sess.run(embedder.loss[val_key],
                #                         feed_dict={inputs: val_inputs, target_dict[
                #                             val_key]:
                #                             val_labels})
                #     print('Validation Loss at step {} key {} : {}'.format(i + 1,
                #                                                           val_key,
                #                                                           val_loss))
                eval(embedder.loss, batch_val, 'Validation', 'Loss')

                # for key, train_input_label in enumerate(batch):
                #     train_key = str(key)
                #     train_inputs = train_input_label[0]
                #     train_labels = train_input_label[1]
                #
                #     train_acc = sess.run(embedder.accuracy[train_key],
                #         {inputs: train_inputs, target_dict[train_key]: train_labels})
                #     print('Training Accuracy at step {} key {} : {}'.format(i +
                #                                                             1,
                #                                                             train_key,
                #                                                             train_acc))

                eval(embedder.accuracy, batch, 'Train', 'Accuracy')

                # for key, val_input_label in enumerate(batch_val):
                #     val_key = str(key)
                #     val_inputs = val_input_label[0]
                #     val_labels = val_input_label[1]
                #
                #     val_acc = sess.run(embedder.accuracy[val_key],
                #                          {inputs: val_inputs,
                #                           target_dict[val_key]: val_labels})
                #
                #     print('Validation Accuracy at step {} key {} : {}'.format(i +
                #                                                             1,
                #                                                             val_key,
                #                                                             val_acc))
                eval(embedder.loss, batch_val, 'Validation', 'Accuracy')


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
