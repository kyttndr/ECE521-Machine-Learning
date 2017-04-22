import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def buildGrapph():
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='bias')

    y_predicted = tf.matmul(X, W) + b

    # Error definition
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predicted, targets=y_target, name='cross_entropy'))
    meanSquaredError = 1.0 / 2.0 * tf.reduce_mean(tf.reduce_sum(tf.square(y_predicted - y_target), reduction_indices=1,
                                                                name='squared_error'), name='mean_squared_error')

    decay_coeff = 0.0
    weight_decay = decay_coeff * tf.nn.l2_loss(W)

    loss = cross_entropy + weight_decay
    #loss = meanSquaredError + weight_decay

    learningRate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss)

    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_predicted > 0, tf.float32), y_target), tf.float32))

    return X, y_target, W, b, y_predicted,loss,train, accuracy


def logisticRegression():
    X, y_target, W, b, y_predicted, loss, train, accuracy = buildGrapph()

    # load data
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]

        posClass = 2
        negClass = 9

        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)

        Target[Target == posClass ] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]

        Data = np.array(np.reshape(Data, (-1, 784)));

        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

    # Initialize session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    # hyper parameters
    B = 500
    iterNum = 2001

    numBatches = np.floor(len(trainData)/B)
    train_loss = []
    acc_list = []
    i = 0
    for step in xrange(0,iterNum):

        ## sample minibatch without replacement
        if i == numBatches :
            randIdx = np.arange(len(trainData))
            np.random.shuffle(randIdx)
            trainData = trainData[randIdx]
            trainTarget = trainTarget[randIdx]
            i = 0

        feeddict = {X: trainData[i * B:(i + 1) * B],
                    y_target: trainTarget[i * B:(i + 1) * B]}


        i += 1

        ## Update model parameters
        _, err = sess.run([train, loss], feed_dict= feeddict)
        acc = accuracy.eval({X: testData, y_target: testTarget})
        train_loss.append(err)
        acc_list.append(acc)

        if not (step % 100) or step < 10:
            print("Iter: %3d, cross-entropy-loss: %4.2f, accuracy is %4.2f "%(step, err,acc))

    #Return the validation accuracy
    print ("Train accuracy is %4.2f" % (accuracy.eval({X: trainData, y_target: trainTarget})))
    print ("Validation accuracy is %4.2f" %(accuracy.eval({X: validData, y_target: validTarget})))
    print ("Test accuracy is %4.2f" % (accuracy.eval({X: testData, y_target: testTarget})))

    #plot the result
    step = np.arange(0, iterNum)
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.title('learning rate = %.4f' % (0.01))
    plt.xlabel('Number of updates')
    plt.ylabel('cross-entropy Loss')
    plt.plot(step, train_loss, '-')

    plt.subplot(2, 1, 2)
    plt.xlabel('Number of updates')
    plt.ylabel('accuracy')
    plt.plot(step, acc_list, '-')
    plt.show()



def CELvsMSE():

    steps = 1000
    y_predict = tf.placeholder(tf.float32, [1,] ,name='y_predict')
    y_target = tf.zeros([1,],tf.float32, name= 'y_target')

    CEL = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict,targets=y_target,name = 'cross_entropy'))
    meanSquaredError = tf.reduce_mean(tf.square(y_predict - y_target), name='mean_squared_error')

    y = np.linspace(0.0,1.0,num=1000)[:,np.newaxis]


    cel = np.zeros((steps,),dtype=np.float32)
    mse = np.zeros((steps,),dtype=np.float32)

    # Initialize session
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    for i in xrange(0,steps):
        cel[i], mse[i] = sess.run([CEL,meanSquaredError],feed_dict = {y_predict : y[i]})

    plt.figure(1)
    plt.title("cross-entropy loss vs squared-error loss")
    plt.xlabel("prediction y")
    plt.ylabel("error")
    plt.plot(y,mse,'r-',label = "mean squared loss")
    plt.plot( y, cel, 'b-', label = "cross-entropy loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #logisticRegression()
    CELvsMSE()

