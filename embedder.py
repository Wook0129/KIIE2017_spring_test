import config
import os
import tensorflow as tf
from batch_generator import TrainValBatchGenerator
from data_handler import DataHandler


class Embedder:

	def __init__(self, *, inputs, target, num_instances, num_of_vars, total_num_of_bins, embedding_size):

		with tf.variable_scope('embedding_layer'):
			self.variable_embeddings =  tf.get_variable(name='variable_embeddings',
															shape=[total_num_of_bins, embedding_size],
																initializer=tf.random_normal_initializer())
			self.instance_embeddings = tf.get_variable(name='instance_embeddings',
															shape=[num_instances, embedding_size],
																initializer=tf.random_normal_initializer())

		with tf.variable_scope('output_layer'):
			softmax_weights = tf.get_variable(name='softmax_weights',
												shape=[embedding_size, total_num_of_bins],
													initializer=tf.truncated_normal_initializer())
			softmax_biases = tf.get_variable(name='softmax_biases',
												shape=[total_num_of_bins],
													initializer=tf.zeros_initializer())

		embed_variable = tf.nn.embedding_lookup(self.variable_embeddings, inputs[:,:num_of_vars-1])
		embed_instance = tf.nn.embedding_lookup(self.instance_embeddings, inputs[:,num_of_vars-1])
		embedding = tf.reduce_sum(embed_variable, reduction_indices=1) + embed_instance

		logits = tf.matmul(embedding, softmax_weights) + softmax_biases
		onehot_target = tf.squeeze(tf.one_hot(indices=target, depth=total_num_of_bins), 1)

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_target))
		self.optimizer = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
		self.predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_target, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.predictions, "float"))
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

# Train Embedding & Prepare for Visualization
def train_embedder(data, *, max_iteration=config.max_iteration,
								train_batch_size=config.train_batch_size,
									val_batch_size=config.val_batch_size,
										embedding_size=config.embedding_size):
	if not os.path.exists(config.LOG_DIR):
		os.makedirs(config.LOG_DIR)
	data_handler = DataHandler(data) # Pandas format data
	data_handler.save_metadata()

	batch_generator = TrainValBatchGenerator(data=data_handler.onehot_encoded_data.as_matrix(),
												train_batch_size=train_batch_size,
													val_batch_size=val_batch_size)
	with tf.Session() as sess:
		inputs = tf.placeholder(tf.int32, shape=[None, data_handler.num_of_vars]) # (num_vars - 1) + 1 instance_id
		target = tf.placeholder(tf.int32, shape=[None, 1])

		with tf.variable_scope('Embedder'):
			embedder = Embedder(inputs=inputs,
								target=target,
								num_instances=data_handler.num_instances,
								num_of_vars=data_handler.num_of_vars,
								total_num_of_bins=data_handler.total_num_of_bins,
								embedding_size=embedding_size)
		sess.run(embedder.init)

		for i in range(max_iteration):
			train_inputs, train_labels = batch_generator.next_train_batch()
			sess.run(embedder.optimizer, feed_dict={inputs : train_inputs, target : train_labels})
			
			if (i+1) % config.print_loss_every == 0:
				train_loss = sess.run(embedder.loss, feed_dict={inputs:train_inputs, target:train_labels})
				print('Training Loss at step {} : {}'.format(i+1, train_loss))
				
				val_inputs, val_labels = batch_generator.next_val_batch()
				val_loss = sess.run(embedder.loss, feed_dict={inputs:val_inputs, target:val_labels})
				print('Validation Loss at step {} : {}'.format(i+1, val_loss))
				
				train_acc = embedder.accuracy.eval({inputs: train_inputs, target: train_labels})
				print('Training Accuracy at step {} : {}'.format(i+1, train_acc))
				
				val_acc = embedder.accuracy.eval({inputs: val_inputs, target: val_labels})
				print('Validation Accuracy at step {} : {}'.format(i+1, val_acc), '\n')

				embedder.saver.save(sess, save_path=os.path.join(config.LOG_DIR, config.model_save_filename), 
										  global_step=(i+1))
		
		# Save And Visualize Embeddings
		tensorboard_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
		var_embedding = tensorboard_config.embeddings.add()
		var_embedding.tensor_name = embedder.variable_embeddings.name
		var_embedding.metadata_path = os.path.join(config.LOG_DIR, config.metadata_filename)
		summary_writer = tf.summary.FileWriter(config.LOG_DIR)
		tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, tensorboard_config)
