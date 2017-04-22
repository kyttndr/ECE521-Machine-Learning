import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from time import time
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

num_epochs = 100
batch_size = 500
unique_name = "s"

def train_neural_network(learning_rate, num_layers, hidden_units, dropout, weight_decay_coefficient):

    global unique_name

    if dropout:
        train_prob = 0.5
    else:
        train_prob = 1.0

    keep_prob = tf.placeholder("float32")

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    hidden_1_w = tf.get_variable(name=unique_name, shape=[784, hidden_units[0]], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    hidden_1_b = tf.Variable(tf.zeros([hidden_units[0]]))
    layer1 = tf.nn.relu(tf.matmul(x, hidden_1_w) + hidden_1_b)
    layer_1_drop = tf.nn.dropout(layer1, keep_prob)

    unique_name += "s"

    w = [hidden_1_w]
    b = [hidden_1_b]
    l = [layer1]
    l_drop = [layer_1_drop]

    for i in range(num_layers - 1):
        w_temp = tf.get_variable(name=unique_name, shape=[hidden_units[i], hidden_units[i + 1]], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b_temp = tf.Variable(tf.zeros([hidden_units[i + 1]]))

        unique_name += "s"

        w.append(w_temp)
        b.append(b_temp)
        
        l.append(tf.nn.relu(tf.matmul(l_drop[-1], w[-1]) + b[-1]))
        l_drop.append(tf.nn.dropout(l[-1], keep_prob))
    
    w_last = tf.get_variable(name=unique_name, shape=[hidden_units[-1], 10], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b_last = tf.Variable(tf.zeros([10]))

    unique_name += "s"

    prediction = tf.matmul(l_drop[-1], w_last) + b_last

    weight_decay_loss = weight_decay_coefficient * reduce(lambda s, t: s + t, map(lambda weight: tf.reduce_sum(tf.square(weight)), w)) / len(w)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y)) + weight_decay_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, "float32"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            
            total_batches = int(len(trainTarget) / batch_size)
            for i in range(total_batches):
                batch_xs = trainData[i * batch_size: (i + 1) * batch_size]
                batch_ys = trainTarget[i * batch_size: (i + 1) * batch_size]
                _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: train_prob})

        cost_valid, accuracy_valid = sess.run([cost, accuracy], feed_dict={x: validData, y: validTarget, keep_prob: 1.0})
        cost_test, accuracy_test = sess.run([cost, accuracy], feed_dict={x: testData, y: testTarget, keep_prob: 1.0})

        print "Learning Rate: %8.6f" %(learning_rate)
        print "Number of Hidden Layers: %3d" %(num_layers)
        temp_s = ""
        for h in hidden_units:
            temp_s += str(h)
            temp_s += ", "
        print "Number of Hidden Units: (%s)" %(temp_s[:-2])
        print "Dropout: %s" %(dropout)
        print "Weight Decay Coefficient: %8.6f" %(weight_decay_coefficient)
        print "Valid Accuracy: %5.2f%%" %(accuracy_valid * 100)
        print "Test Accuracy: %5.2f%%" %(accuracy_test * 100)

curr_time = time()

random.seed(curr_time)

log_learning_rate = [random.uniform(-7.5, -4.5) for i in range(5)]

learning_rate = np.exp(log_learning_rate).astype('float32')

log_weight_decay_coefficient = [random.uniform(-9, -6) for i in range(5)]

weight_decay_coefficient = np.exp(log_weight_decay_coefficient).astype('float32')

num_layers = [random.randint(1, 5) for i in range(5)]

hidden_units = []

for i in range(5):
    temp_hidden_units = [random.randint(100, 500) for i in range(num_layers[i])]
    hidden_units.append(temp_hidden_units)

dropout = [(random.randint(0, 1) == 1) for i in range(5)]

cnt = 1
for arg_array in zip(learning_rate, num_layers, hidden_units, dropout, weight_decay_coefficient):
    print "Mode%d: " %(cnt) 
    train_neural_network(*arg_array)
    print "\n"
    cnt += 1













