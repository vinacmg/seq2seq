import numpy as np
import tensorflow as tf


class Seq2Seq(object):

	def __init__(self, input_vocab_size, target_vocab_size, buckets, 
		hidden_units, num_layers, batch_size, learning_rate, 
		enc_seq_lenght, dec_seq_lenght, embeddings, encoder_input_ids,
		decoder_input_ids,decoder_target_ids):
		#
		#
		#
		#
		#self.encoder_inputs, self.decoder_inputs = [max-time, batch-size, ...]
		#
		#
		self.encoder_inputs = tf.nn.embedding_lookup(embeddings, encoder_input_ids) 
		self.decoder_inputs = tf.nn.embedding_lookup(embeddings, decoder_input_ids)

		#Encoder######################

		cell = tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True) #como inicializar weights??
		encoder_cell = cell
		if num_layers > 1:
			encoder_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

		self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
			cell=encoder_cell,
			dtype=tf.float32,
			sequence_length=enc_seq_lenght,
			inputs=self.encoder_inputs,
			time_major=True)

		###############################

		#Decoder#######################

		# lembrar de encoder_final_state.h 

		#weights of output projection
		W = tf.Variable(tf.random_uniform([hidden_units, target_vocab_size], -1, 1), dtype=tf.float32)
		#biases of output projection
		b = tf.Variable(tf.zeros([target_vocab_size]), dtype=tf.float32)

		decoder_cell = cell
		if num_layers > 1:
			decoder_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

		self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
			cell=decoder_cell,
			dtype=tf.float32,
			sequence_length=dec_seq_lenght,
			inputs=self.decoder_inputs,
			time_major=True)

		###############################

		decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
		decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
		decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
		decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, target_vocab_size))

		#para inferencia###################
		#decoder_probabilites = tf.nn.softmax(decoder_logits)
		#decoder_prediction = tf.argmax(decoder_probabilites, 2) #[max-time,batch-size]
		###################################

		stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(decoder_target_ids, depth=target_vocab_size, dtype=tf.float32),
			logits=decoder_logits,)

		loss = tf.reduce_mean(stepwise_cross_entropy)
		train_op = tf.train.AdamOptimizer().minimize(loss)

		sess.run(tf.global_variables_initializer())
