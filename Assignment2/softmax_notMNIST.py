import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def logisticRegression():
    # load data
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx] / 255.
    Data = np.reshape(Data,(-1,784))

    Target = Target[randIndx]
    zero = np.zeros((len(Target), 10))
    for i in range(0, len(Target)):
        zero[i, Target[i]] = 1
    Target = zero

    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]




    X = tf.placeholder(tf.float32, [None,784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 10], name='target_y')
    W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.5), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')

    y_predicted = tf.matmul(X, W) + b

    # Error definition
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target,logits=y_predicted,name = 'cross_entropy'))

    decay_coeff = 0.01
    weight_decay = decay_coeff * tf.nn.l2_loss(W)

    loss = cross_entropy + weight_decay

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learningRate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
    train = optimizer.minimize(loss)

    # Initialize session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    # hyper parameters
    B = 500
    iterNum = 3001

    numBatches = np.floor(len(trainData)/B)
    err = np.zeros(shape=(iterNum,), dtype=np.float32)
    acc = np.zeros(shape=(iterNum,), dtype=np.float32)

    for step in xrange(0,iterNum):
        ## sample minibatch without replacement
        if step % numBatches ==0 :
            randIdx = np.arange(len(trainData))
            np.random.shuffle(randIdx)
            trainData = trainData[randIdx]
            trainTarget = trainTarget[randIdx]
            i = 0

        x_batch = trainData[i*B:(i+1)*B]
        y_batch = trainTarget[i*B:(i+1)*B]

       ## assert (x_batch.shape == (500,784))

        i += 1

        ## Update model parameters
        _, err[step], currentW, currentb, yhat = sess.run([train, loss, W, b, y_predicted],
                                                          feed_dict= {X: x_batch, y_target:y_batch})

        #print(yhat)
        #print(y_batch)

        acc[step] = accuracy.eval(feed_dict={X: testData,
                                        y_target: testTarget})

        #print (yhat)
        if not (step % 100) or step < 10:
            print("Iter: %3d, cross-entropy-loss: %4.2f, accuracy is %4.2f "%(step, err[step],acc[step]))

    #plot the result
    step = np.arange(0, iterNum)
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title('learning rate = %.4f' % (learningRate))
    plt.xlabel('Number of updates')
    plt.ylabel('cross-entropy Loss')
    plt.plot(step, err, '-')

    plt.subplot(2, 1, 2)
    plt.xlabel('Number of updates')
    plt.ylabel('accuracy')
    plt.plot(step, acc, '-')
    plt.show()


if __name__ == '__main__':
    logisticRegression()
