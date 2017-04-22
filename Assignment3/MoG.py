import tensorflow as tf
import numpy as np
import math
import utils
import matplotlib.pyplot as plt

def logPosterior(X,u,stddev):
    '''

    :param X: B * D matrix with B number of D dimensions data
    :param u: Centroids
    :param stddev: the standard deviation for multivariate Gaussian
    :return: log probability for all pairs of B data points and K clusters
    '''
    distances = tf.reduce_sum((tf.expand_dims(u, 2)-
                               tf.expand_dims(tf.transpose(X), 0)) ** 2, 1)
    #var = tf.sqr(stddev)
    var = tf.exp(stddev)
    return - (0.5 * tf.cast(tf.rank(X),tf.float32) * tf.log(2 * math.pi * var) + 0.5 * (tf.multiply(1/var, distances)))


def logProbZgivenX (X, u, Pz, stddev):
    '''

    :param X:  B * D matrix with B number of D dimensions data points
    :param u:  Centroids
    :param Pz: Probability for each latent variable
    :param stddev: standard deviation for every dimension in multivariate Gaussian
    :return: the log probability for cluster variable z given the data vector: log(P(z|x)
    '''

    logGaussian = logPosterior(X,u,stddev)
   #logPz = tf.log(Pz)
    logPz = utils.logsoftmax(Pz)
    logSumExp = utils.reduce_logsumexp(logGaussian + logPz, 0)

    return logGaussian + logPz - logSumExp

def logLikelihood(X, u, Pz, stddev):
    '''

    :param X: B * D matrix with B number of D dimensions data points
    :param u: Centroids
    :param Pz: Probability for each latent variable
    :param stddev: standard deviation for every dimension in multivariate Gaussian
    :return: the log likelihood of X
    '''

    logGaussian = logPosterior(X, u, stddev)
 #   logPz = tf.log(Pz)
    logPz = utils.logsoftmax(Pz)
    logSumExp = utils.reduce_logsumexp(logGaussian + logPz, 0)


    return tf.reduce_sum(logSumExp, 0)


# load data
data = np.load("data2D.npy")
#data = np.load("data100D.npy")

def buildGrapph(n_clusters):
    X = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]], name='input_x')

    u = tf.Variable(tf.truncated_normal(shape=[n_clusters, data.shape[1]], mean=np.mean(data), stddev=0.5, dtype=tf.float32),
                    name='centroids')
    dev = tf.Variable(tf.truncated_normal(shape=[n_clusters, 1], stddev=0.5, dtype=tf.float32),name = 'standard_deviance')
    Pz = tf.Variable(tf.ones(shape = [n_clusters, 1], dtype=tf.float32)/n_clusters, name='latent_variable_probability')

    distances = logPosterior(X, u, dev)

    y_predict = tf.argmax(distances,0)

    loss = -logLikelihood(X, u, Pz, dev)


    learningRate = 0.1
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate,beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss)


    return X, u, dev, Pz, y_predict, loss, train


def MoG(data):
    #num_clusters = [7, 8, 9, 10, 11, 12]
    num_clusters = [1,2,3,4,5]
    #num_clusters = [3]

    np.random.seed(521)
    randIndx = np.arange(len(data))
    np.random.shuffle(randIndx)
    data = data[randIndx]
    valid_data = data[0:len(data) / 3]

    for K in num_clusters:
        X, u, dev, Pz, y_predict, loss, train = buildGrapph(K)

        # Initialize session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)

        # hyper parameters
        iterNum = 501

        train_loss = []
        for step in xrange(0,iterNum):

            ## sample minibatch without replacemen
            feeddict = {X: data}
        #    feeddict = {X: valid_data}

            ## Update model parameters
            _, err, centroids, y, phi, psi = sess.run([train, loss, u, y_predict, dev, Pz], feed_dict = feeddict)
            train_loss.append(err)

            if not (step % 100) or step < 10:
                print("Iter: %3d, cross-entropy-loss: %4.2f"%(step, err))

        print("The error is %4.2f for K =%d"%(err,K))

        '''
        print (" For K = %d , the best model parameters learnt: " %K)
        print (" Mean is ")
        print (centroids)
        print (" Phi is")
        print (phi)
        print ("Psi is")
        print (psi)
        '''
        plotLoss(train_loss, K)
    plotKmeans(K, centroids,y, valid_data)

    plt.show()



def plotKmeans(K, centroids, y, data):
    plt.figure(K)
    plt.title("Scatter plot for K = %d" % (K))
    #    color = np.array(['ro','bo', 'go'])
    percent = np.zeros(shape=(K,))
    for i in xrange(0, len(data)):
        percent[y[i]] += 1

    color = plt.cm.rainbow(np.linspace(0, 1, K))
    plt.scatter(data[:, 0], data[:, 1], c=color[y[:]], marker='o')
    plt.plot(centroids[:, 0], centroids[:, 1], 'yo', markersize=10)

    print("The percentage of data points belonging to each of the K clusters :")
    for i in xrange(0, K):
        print("K = %d : %4.2f " % (K, percent[i] / len(data)))


def plotLoss(loss, K):
    steps = np.arange(0,len(loss))
    plt.figure(1)
    plt.title("MoG loss vs updates when K = %d"%(K))
    plt.plot(steps, loss)
    plt.show()


if __name__ == '__main__':
    MoG(data)
    '''
    plt.figure(1)
    plt.title("Number of clusters vs Validation loss")
    K = [1,2,3,4,5]
    loss = [11620.37,5868.02,2083.17,1289.72,519.19]
    plt.plot(K,loss)
    plt.show()
    '''
