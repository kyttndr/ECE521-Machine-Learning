import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Layer_Wise_Building_Block import layer_wise_building_block

#load data
with np.load("notMNIST.npz") as data:
    Data, Target = data["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx] / 255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

trainData = trainData.reshape((trainData.shape[0], 784))
validData = validData.reshape((validData.shape[0], 784))
testData = testData.reshape((testData.shape[0], 784))

def labels_one_hot(target):
    num_examples = target.size
    labels_one_hot = np.zeros((num_examples, target.max() - target.min() + 1))
    labels_one_hot[np.arange(num_examples), target.ravel()] = 1
    return labels_one_hot

trainTarget = labels_one_hot(trainTarget)
validTarget = labels_one_hot(validTarget)
testTarget = labels_one_hot(testTarget)

hidden_units = 1000
learning_rate = 1e-2
num_epochs = 100
batch_size = 500

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

def neural_network_model(data):
    hidden_1_w = tf.get_variable("hidden_Weight_1", [784, hidden_units], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    hidden_1_b = tf.Variable(tf.zeros([hidden_units]))
    layer1 = tf.nn.relu(tf.matmul(data, hidden_1_w) + hidden_1_b)
    hidden_2_w = tf.get_variable("hidden_Weight_2", [hidden_units, 10], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    hidden_2_b = tf.Variable(tf.zeros([10]))
    output = tf.matmul(layer1, hidden_2_w) + hidden_2_b
    return output, hidden_1_w, hidden_2_w

def train_neural_network(output):

    global trainTarget, trainData, validData, validTarget, testData, testTarget

    prediction, w1, w2 = neural_network_model(output)
    weight_decay_loss = 3.0 * tf.pow(10.0, -4.0) * (tf.reduce_sum(tf.square(w1)) + tf.reduce_sum(tf.square(w2))) / 2.0

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y)) + weight_decay_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, "float32"))

    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    train_cost = []
    valid_cost = []
    test_cost = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            randIndx = np.arange(len(trainData))
            np.random.shuffle(randIndx)
            trainData = trainData[randIndx]
            trainTarget = trainTarget[randIndx]

            total_batches = int(len(trainTarget) / batch_size)
            for i in range(total_batches):
                batch_xs = trainData[i * batch_size: (i + 1) * batch_size]
                batch_ys = trainTarget[i * batch_size: (i + 1) * batch_size]
                _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_xs, y: batch_ys})

            cost_train, accuracy_train = sess.run([cost, accuracy], feed_dict={x: trainData, y: trainTarget})
            cost_valid, accuracy_valid = sess.run([cost, accuracy], feed_dict={x: validData, y: validTarget})
            cost_test, accuracy_test = sess.run([cost, accuracy], feed_dict={x: testData, y: testTarget})

            train_cost.append(cost_train)
            train_accuracy.append(accuracy_train)
            valid_cost.append(cost_valid)
            valid_accuracy.append(accuracy_valid)
            test_cost.append(cost_test)
            test_accuracy.append(accuracy_test)
            
            if epoch % 10 == 9:
                print "Epoch: %3d" % (epoch + 1)
                print "\tTraining Set:   Cost: %8.3f Accuracy: %d%%" %(cost_train, accuracy_train * 100)
                print "\tValid Set:      Cost: %8.3f Accuracy: %d%%" %(cost_valid, accuracy_valid * 100)
                print "\tTest Set:       Cost: %8.3f Accuracy: %d%%\n\n" %(cost_test, accuracy_test * 100)
            
        print "Optimization Finished!"
        print "Test Cost: ", cost_test
        print "Test Accuracy: ", accuracy_test
        train_error = [1 - i for i in train_accuracy]
        valid_error = [1 - i for i in valid_accuracy]
        test_error = [1 - i for i in test_accuracy]

        plt.figure(1)
        plt.title("Error vs Number of Epochs")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Error")
        plt.plot(train_error)
        plt.plot(valid_error)
        plt.plot(test_error)
        plt.legend(['Training ', 'Validation', 'Test'])
        plt.grid()

        plt.figure(2)
        plt.title("Loss vs Number of Epochs")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.plot(train_cost)
        plt.plot(valid_cost)
        plt.plot(test_cost)
        plt.legend(['Training ', 'Validation', 'Test'])
        plt.grid()
        plt.show()

train_neural_network(x)













