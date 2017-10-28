import numpy as np
import tensorflow as tf

input_max_lenght = 5
encoder_inputs = tf.placeholder(tf.float32, [input_max_lenght, None, 1]) #seq max lenght, batch size, data dimension size

encoder_hidden_num = 5
encoder_outputs_size = 5
encoder_seq_lenght = tf.placeholder(tf.int32, [None])


'''
class Seq2Seq(object)

	def __init__():
'''

class BidirectionalEncoder(object):

	def __init__(self, encoder_inputs, input_max_lenght, encoder_hidden_num, encoder_seq_lenght):

		encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units = encoder_hidden_num, state_is_tuple = True)

		((encoder_outputs_fw, encoder_outputs_bw), (encoder_final_state_fw, encoder_final_state_bw)) = tf.nn.bidirectional_dynamic_rnn( #time_major = true -> output_fw = [max_time, batch_size, cell_fw.output_size]
			cell_fw = encoder_cell,
			cell_bw = encoder_cell,
			dtype = tf.float32,
			sequence_length = encoder_seq_lenght,
			inputs = encoder_inputs,
			time_major = True) 

		self.encoder_outputs = tf.concat((encoder_outputs_fw, encoder_outputs_bw), 2, name = 'encoder_outputs')

		encoder_final_state_c = tf.concat(
		    (encoder_final_state_fw.c, encoder_final_state_bw.c), 1, name = 'encoder_final_state_c')

		encoder_final_state_h = tf.concat(
		    (encoder_final_state_bw.h, encoder_final_state_bw.h), 1, name = 'encoder_final_state_h')

		#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
		self.encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
		    c = encoder_final_state_c,
		    h = encoder_final_state_h
		)

encoder =  BidirectionalEncoder(encoder_inputs, input_max_lenght, encoder_hidden_num, encoder_seq_lenght)