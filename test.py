import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

max_time = 5
data_size = 20000
target_vocab_size = 8
embedding_size = 6
hidden_units = 20
num_layers = 2
batch_size = 100
epochs = 5


def generate_data(x_size, y_size):
	return np.random.randint(2, 8, size=(x_size,y_size))


EOS = np.matrix(np.ones(data_size))
x = generate_data(max_time,data_size)
y = np.concatenate((EOS, x),)
target = np.concatenate((x, EOS),)
enc_seq_length = np.array([5]*batch_size)
dec_seq_length = np.array([6]*batch_size)

sess = tf.Session()
loader = tf.train.import_meta_graph('save/my_test_model.meta')
loader.restore(sess, tf.train.latest_checkpoint('save/'))

encoder_input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')
encoder_sequence_length = tf.placeholder(shape=[None], dtype=tf.int32, name='encoder_sequence_length')
decoder_input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')
decoder_sequence_length = tf.placeholder(shape=[None], dtype=tf.int32, name='decoder_sequence_length')
decoder_targets_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')

embeddings = tf.Variable(tf.random_uniform([target_vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs = tf.nn.embedding_lookup(embeddings, encoder_input_ids) 
decoder_inputs = tf.nn.embedding_lookup(embeddings, decoder_input_ids)

#Encoder######################

#como inicializar weights??

if num_layers > 1:
	stacked_encoder = []
	for layer in range(num_layers):
		stacked_encoder.append(tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True))
	encoder_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_encoder, state_is_tuple=True)
else: encoder_cell = tf.nn.rnn_cell.MultiRNNCell(tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True), state_is_tuple=True)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
	cell=encoder_cell,
	dtype=tf.float32,
	sequence_length=encoder_sequence_length,
	inputs=encoder_inputs,
	time_major=True,
	scope="encoder")

###############################

#Decoder#######################

#weights of output projection
W = tf.Variable(tf.random_uniform([hidden_units, target_vocab_size], -1, 1), dtype=tf.float32)
#biases of output projection
b = tf.Variable(tf.zeros([target_vocab_size]), dtype=tf.float32)

if num_layers > 1:
	stacked_decoder = []
	for layer in range(num_layers):
		stacked_decoder.append(tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True))
	decoder_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_decoder, state_is_tuple=True)
else: decoder_cell = tf.nn.rnn_cell.MultiRNNCell(tf.nn.rnn_cell.LSTMCell(num_units = hidden_units, state_is_tuple=True), state_is_tuple=True)


decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
	cell=decoder_cell,
	dtype=tf.float32,
	sequence_length=decoder_sequence_length,
	initial_state=encoder_final_state,
	inputs=decoder_inputs,
	time_major=True,
	scope="decoder")

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
	labels=tf.one_hot(decoder_targets_ids, depth=target_vocab_size, dtype=tf.float32),
	logits=decoder_logits,)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

###########################################
loss_track = []

for epoch in range(0, epochs):
	for batch in range(0, 199):
		input_x = x[:,batch*batch_size:(batch+1)*batch_size]
		input_y = y[:,batch*batch_size:(batch+1)*batch_size]
		target_y = target[:,batch*batch_size:(batch+1)*batch_size]

		_, l = (sess.run([train_op, loss], feed_dict={encoder_input_ids: input_x, 
			decoder_input_ids: input_y, 
			encoder_sequence_length: enc_seq_length, 
			decoder_targets_ids: target_y,
			decoder_sequence_length: dec_seq_length}))
		loss_track.append(l)
'''
input_x = x[:,19900:20000]
input_y = y[:,19900:20000]
target_y = target[:,19900:20000]

decoder_probabilites = tf.nn.softmax(decoder_logits)
decoder_prediction = tf.argmax(decoder_probabilites, 2) #[max-time,batch-size]

prediction = sess.run(decoder_prediction, feed_dict={encoder_input_ids: input_x, 
			decoder_input_ids: input_y, 
			encoder_sequence_length: enc_seq_length, 
			decoder_targets_ids: target_y,
			decoder_sequence_length: dec_seq_length})

for i in range(100):
	print(input_y[:,i], end= " ")
	print(prediction[:,i])

plt.plot(loss_track)
plt.show()

print(loss_track[-1])
'''

#saver.save(sess, "save/my_test_model")

def inference(decoder_logits):

	while (True):
		lista = []
		inp = input("Sequence: ")
		for char in inp:
			lista.append(int(char))

		input_x = np.transpose(np.matrix(lista))
		input_y = np.transpose(np.matrix([1] + lista)) #dont want to change lista values
		target_y = np.transpose(np.matrix(lista + [1]))
		enc_seq_length = np.array([len(input_x)])
		dec_seq_length = np.array([len(target_y)])

		decoder_probabilites = tf.nn.softmax(decoder_logits)
		decoder_prediction = tf.argmax(decoder_probabilites, 2) #[max-time,batch-size]

		prediction = sess.run(tf.transpose(decoder_prediction), feed_dict={encoder_input_ids: input_x, 
				decoder_input_ids: input_y, 
				encoder_sequence_length: enc_seq_length, 
				decoder_targets_ids: target_y,
				decoder_sequence_length: dec_seq_length})

		print(prediction)

inference(decoder_logits)