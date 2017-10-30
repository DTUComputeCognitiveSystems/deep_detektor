
# Import
import tensorflow as tf
import numpy as np


class MyLogisticRegression():

    def __init__(self):

        self.learning_rate = 0.001

    def fit(self, data, sess):
        #yy = np.array(data['labels']).astype(float)

        num_features = data['bow'].shape[1]

        # tf Graph Input
        self.x = tf.placeholder(tf.float32, [None, num_features])
        self.y = tf.placeholder(tf.float32, [None,])  # binary classfication

        # Set model weights
        self.W = tf.Variable(tf.zeros([num_features, 1]))
        self.b = tf.Variable(tf.zeros([1]))

        # Construct model
        self.pred = tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.b)  # sigmoid

        # Minimize error using cross entropy
        #self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))
        self.cost = tf.reduce_mean(-self.y * tf.log(self.pred))

        # Gradient Descent
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        training_epochs = 5
        display_step = 1


        # Initialize the variables (i.e. assign their default value)
        self.initializer = tf.global_variables_initializer()


        ## Start training
        # Run the initializer
        sess.run(self.initializer)

        # Training cycle
        for epoch in range(training_epochs):
            _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: data['bow'].todense(),
                                                          self.y: data['labels']})
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")


    def predict(self, data, sess):
        #yy = np.array(data['labels']).astype(float)
        ll = sess.run(self.pred, feed_dict={self.x: data['bow'].todense()})
        #print(ll)
        #print()
        return ll<0.5