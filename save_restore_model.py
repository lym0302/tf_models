# coding=utf-8

'''
liangym   20190213
learning from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/save_restore_model.py
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import tensorflow as tf

# Step 1 --> load data -- Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "./logir/model.ckpt"

# Network Parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Step 2 --> Create model
def multilayer_perceptron(x, weights, biases):
  # Hidden layer with Relu activation
  layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
  layer_1 = tf.nn.relu(layer_1)

  layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  layer_2 = tf.nn.relu(layer_2)

  # Output layer with linear activation
  out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
  return out_layer

# Store layer weights & biases
weights = {
  'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
  'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
  'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
  'b1':tf.Variable(tf.random_normal([n_hidden_1])),
  'b2':tf.Variable(tf.random_normal([n_hidden_2])),
  'out':tf.Variable(tf.random_normal([n_classes]))
}

# Step 3 --> Construct model and Define loss and optimizer
pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize the variable
init = tf.global_variables_initializer()

# Running first session
print("Starting 1st session...")
with tf.Session() as sess:
  sess.run(init)

  # Step 4 --> training
  for epoch in range(3):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

      avg_cost += c / total_batch

    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
  print("First Optimization Finished!")

  # test
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

  # Step 5 --> Save model weights to disk
  saver = tf.train.Saver()
  save_path = saver.save(sess, model_path)
  print("Model saved in file: %s" % save_path)



# Running a new session
print("Starting 2nd session...")
with tf.Session() as sess:
  # Initialize variables
  sess.run(init)

  # Step 6 --> Restore model weights from previously saved model
  saver.restore(sess,model_path)
  print("Model restored from file: %s" % save_path)

  # Resume training
  for epoch in range(7):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
      avg_cost += c / total_batch

    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
  print("Second Optimization Finished!")

  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


