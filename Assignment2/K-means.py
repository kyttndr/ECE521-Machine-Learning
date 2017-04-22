import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.load("data2D.npy")
#data = np.load("data100D.npy")

def buildGrapph(n_clusters):
    X = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]], name='input_x')

    u = tf.Variable( tf.truncated_normal(shape=[n_clusters, data.shape[1]], mean=np.mean(data), stddev=0.5, dtype=tf.float32), name='centroids')
    distances = tf.reduce_sum(tf.square(tf.sub(X[tf.newaxis,:,:],u[:,tf.newaxis,:])), 2)
    y_predict = tf.argmin(distances,0)
    loss = tf.reduce_sum(tf.reduce_min(distances,0),0)


    learningRate = 0.1
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate,beta1=0.9, beta2=0.99, epsilon=1e-5)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    train = optimizer.minimize(loss)


    return X, u, y_predict, loss, train


def K_means():
    num_clusters = [1,2,3,4,5,6,7,8]
#    num_clusters = [3]
    for K in num_clusters:
        X, u, y_predict, loss, train = buildGrapph(K)

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


            ## Update model parameters
            _, err, centroids, y = sess.run([train, loss, u, y_predict], feed_dict= feeddict)
            train_loss.append(err)

            if not (step % 100) or step < 10:
                print("Iter: %3d, cross-entropy-loss: %4.2f"%(step, err))

        print ("Loss for K=%d  is  %4.2f"%(K,err))

        plotLoss(train_loss,K)
    plotKmeans(K,centroids,y)


def plotKmeans(K, centroids, y):

    plt.figure(K)
    plt.title("Scatter plot for K = %d"%(K))
    #    color = np.array(['ro','bo', 'go'])
    percent = np.zeros(shape=(K,))
    for i in xrange (0, len(data)) :
        percent[y[i]] += 1

    color = plt.cm.rainbow(np.linspace(0, 1, K))
    plt.scatter(data[:, 0], data[:, 1], c=color[y[:]], marker='o')
    plt.plot(centroids[:,0],centroids[:,1], 'yo', markersize=10)

    print("The percentage of data points belonging to each of the K clusters :")
    for i in xrange(0 , K):
        print("K = %d : %4.2f " %(K, percent[i]/len(data)))

    plt.show()

def plotLoss(loss, K):
    steps = np.arange(0,len(loss))
    plt.figure(1)
    plt.title("K-means loss vs updates when K = %d"%(K))
    plt.plot(steps, loss)
    plt.show()


def K_means_valid(data):
    np.random.seed(521)
    randIndx = np.arange(len(data))
    np.random.shuffle(randIndx)
    data = data[randIndx]
    valid_data = data[0:len(data)/3]

    num_clusters = [1,2,3,4,5]
    for K in num_clusters:
        X, u, y_predict, loss, train = buildGrapph(K)

        # Initialize session
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)

        # hyper parameters
        iterNum = 401

        train_loss = []
        for step in xrange(0,iterNum):

            ## sample minibatch without replacemen
            feeddict = {X: valid_data}


            ## Update model parameters
            _, err, centroids, y = sess.run([train, loss, u, y_predict], feed_dict= feeddict)
            train_loss.append(err)

            if not (step % 100) or step < 10:
                print("Iter: %3d, cross-entropy-loss: %4.2f"%(step, err))


if __name__ == '__main__':
    #K_means_valid(data)
    K_means()
    '''
    plt.figure(1)
    plt.title("Number of clusters vs Validation loss")
    K = [1,2,3,4,5]
    loss = [12750.89,2998.74,1654.21,1103.08,936.87]
    plt.plot(K,loss)
    plt.show()
    '''
