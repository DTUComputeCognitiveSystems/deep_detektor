import numpy as np
import tensorflow as tf

from models.model_base import DetektorModel


class LogisticRegression(DetektorModel):
    @classmethod
    def name(cls):
        return "LogisticRegression"

    def __init__(self, learning_rate=0.001, training_epochs=10, verbose=False):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.verbose = verbose

    def fit(self, tensor_provider, train_idx, sess, indentation=0):
        # Get training data
        data = tensor_provider.load_data_tensors(train_idx,
                                                 word_embedding=False,
                                                 char_embedding=False,
                                                 pos_tags=False)

        # Get BoW features
        num_features = data['bow'].shape[1]

        # TODO: Model definition and fields should be created in initializer (Python standard)

        # tf Graph Input
        self.x = tf.placeholder(tf.float32, [None, num_features])
        self.y = tf.placeholder(tf.float32, [None, ])  # binary classfication

        # Set model weights
        self.W = tf.Variable(tf.zeros([num_features, 1]))
        self.b = tf.Variable(tf.zeros([1]))

        # Construct model
        self.pred = tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.b)  # sigmoid

        # Minimize error using cross entropy
        self.cost = tf.reduce_mean(-self.y * tf.log(self.pred) - (1 - self.y) * tf.log(1 - self.pred))

        # Gradient Descent
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        display_step = 1

        # Initialize the variables (i.e. assign their default value)
        self.initializer = tf.global_variables_initializer()

        # # Start training
        # Run the initializer
        sess.run(self.initializer)

        # Data
        x = data['bow']
        if not isinstance(x, np.ndarray):
            x = x.todense()
        y = data['labels']

        # Training cycle
        for epoch in range(self.training_epochs):
            _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: x,
                                                                    self.y: y})
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0 and self.verbose:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        print(indentation * " " + "Optimization Finished!")

    def predict(self, data, sess):
        x = data['bow']
        if not isinstance(x, np.ndarray):
            x = x.todense()
        ll = sess.run(self.pred, feed_dict={self.x: x})
        return ll > 0.5
