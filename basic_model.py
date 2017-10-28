import numpy as np
import tensorflow as tf


class Model(object):

	def: _seq2seq_(encoder_inputs, decoder_inputs, feed_previous):
		tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
			encoder_inputs=encoder_inputs,
			decoder_inputs=decoder_inputs,
			self.cell=cell,
			)