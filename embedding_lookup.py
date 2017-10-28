import tensorflow as tf
import numpy as np

sess = tf.Session()

embeddings = tf.Variable([[1,4,7],[2,5,8]])
ids = tf.Variable([[1,1],[0,2],[1,1],[1,0]])
decoder_logits = tf.Variable([[[0.2,0.3,0.05],[0.1,0.3,0.6]],[[0.2,0.5,0.05],[0.4,0.1,0.1]]])

encoder_input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32, name='encoder_inputs')
x = np.random.randint(6, size=(5,10),)
loss = tf.reduce_sum(encoder_input_ids)

sess.run(tf.global_variables_initializer())

#print(tf.nn.embedding_lookup(embeddings, ids))
print(sess.run(encoder_input_ids, feed_dict={encoder_input_ids: x}))
print(x)
