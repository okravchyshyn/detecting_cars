import cv2
import numpy as np
import glob
import pdb
import tensorflow as tf


PATH_TO_CAR_TRAIN = '/home/okravchyshyn/train_car_images/car*.jpg'
PATH_TO_NOCAR_TRAIN = '/home/okravchyshyn/train_nocar_images/nocar*.jpg'

SQURE_SIZE = 64

TESTING_RATE = 0.1

training_data = np.array([])
training_result = np.array([])
testing_data = np.array([])
testing_result = np.array([])


number_of_training_samples = 0 
number_of_testing_samples = 0

n_classes = 2
n_channels = 1

training_samples = 0

BATCH_SIZE = 25
learning_rate = 0.001
training_iters = 100000

def load_training_data():

    global training_data
    global training_result

    global testing_data
    global testing_result

    global number_of_training_samples
    global number_of_testing_samples
    
    img_buffer = np.zeros((SQURE_SIZE, SQURE_SIZE, n_channels), dtype = np.float32) 

    car_files = glob.glob(PATH_TO_CAR_TRAIN)
    nocar_files = glob.glob(PATH_TO_NOCAR_TRAIN)
  
    number_of_all_samples = len(car_files) + len(nocar_files)

    merged_files = np.concatenate((car_files, nocar_files))
    merged_results = np.zeros(number_of_all_samples)
    merged_results[:len(car_files)] = 1

    random_all_sample_idx = np.arange(number_of_all_samples)
    np.random.shuffle(random_all_sample_idx)

    number_of_training_samples = int( len(random_all_sample_idx) * (1.0 - TESTING_RATE) )
    number_of_testing_samples =  len(random_all_sample_idx) - number_of_training_samples


    training_data = np.zeros((number_of_training_samples , SQURE_SIZE, SQURE_SIZE, n_channels), np.float32)
    training_result = np.zeros(number_of_training_samples)

    testing_data = np.zeros((number_of_testing_samples , SQURE_SIZE, SQURE_SIZE, n_channels), np.float32)
    testing_result = np.zeros(number_of_testing_samples )


    idx = 0

    for i in range(number_of_training_samples):

        idx_in_car_files  =  random_all_sample_idx[i]
        f = merged_files[idx_in_car_files]
        r = merged_results[idx_in_car_files]
        print idx_in_car_files, " " , f, " result = " , r 
        img = cv2.imread(f)
        if n_channels == 1:
             gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             img_buffer = gray_image.reshape(gray_image.shape[0], gray_image.shape[1], 1).astype(np.float)/255 
        else:
             img_buffer = img.astype(np.float32)/255

        training_data[i] = img_buffer
        training_result[i] = r

    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print 'Testing data'
    print ''

    for j in  range(number_of_testing_samples) :
        i = j + number_of_training_samples

        idx_in_car_files  =  random_all_sample_idx[i]
        f = merged_files[idx_in_car_files]
        r = merged_results[idx_in_car_files]
        print idx_in_car_files, " " , f, " result = " , r 
        img = cv2.imread(f)
        if n_channels == 1:
             gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             img_buffer = gray_image.reshape(gray_image.shape[0], gray_image.shape[1], 1).astype(np.float)/255 
        else:
             img_buffer = img.astype(np.float32)/255

        testing_data[j] = img_buffer
        testing_result[j] = r


def get_training_batch():
    batch_data = np.zeros((BATCH_SIZE, SQURE_SIZE, SQURE_SIZE, n_channels), np.float32)
    batch_result = np.zeros((BATCH_SIZE, n_classes ))
    begin_from_idx = np.random.randint(number_of_training_samples)  

    for i in range(BATCH_SIZE):
        j = (begin_from_idx + i) % number_of_training_samples
        batch_data[i] = training_data[j]
        if training_result[j] == 1.0:
            batch_result[i] = (1.0, 0.0)
        else:
            batch_result[i] = (0.0, 1.0)


    return batch_data, batch_result


def get_testing_batch():
    batch_data = np.zeros((BATCH_SIZE, SQURE_SIZE, SQURE_SIZE, n_channels), np.float32)
    batch_result = np.zeros((BATCH_SIZE, n_classes))
    begin_from_idx = np.random.randint(number_of_testing_samples)  

    for i in range(BATCH_SIZE):
        j = (begin_from_idx + i) % number_of_testing_samples
 
        batch_data[i] = testing_data[j]
        if training_result[j] == 1.0:
            batch_result[i] = (1.0, 0.0)
        else:
            batch_result[i] = (0.0, 1.0)

    return batch_data, batch_result


load_training_data()
_,_ = get_testing_batch()

pdb.set_trace()
data, results = get_training_batch()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# print tensorflow


def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
   return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, SQURE_SIZE, SQURE_SIZE, n_channels ])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf. placeholder(tf.float32)


wc1 = tf.Variable(tf.random_normal([5, 5, n_channels, 32]))
bc1 = tf.Variable(tf.random_normal([32]))

conv1 = conv2d(x, wc1, bc1)
conv1_with_pooling = max_pool(conv1, k=2)
conv1_with_prob  = tf.nn.dropout(conv1_with_pooling, keep_prob)

wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
bc2 = tf.Variable(tf.random_normal([64]))

conv2 = conv2d(conv1_with_prob, wc2, bc2)
conv2_with_pooling = max_pool(conv2, k=2)
conv2_with_prob = tf.nn.dropout(conv2_with_pooling, keep_prob)

wd1 = tf.Variable(tf.random_normal([16 * 16 * 64, 1024]))
bd1 = tf.Variable(tf.random_normal([1024]))

dense1 = tf.reshape(conv2_with_prob, [-1, wd1.get_shape().as_list()[0]])
dense1_relu = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1))

wout = tf.Variable(tf.random_normal([1024, n_classes]))
bout = tf.Variable(tf.random_normal([n_classes]))

pred = tf.add(tf.matmul(dense1_relu, wout), bout)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y)
cost = tf.reduce_mean(entropy)


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

step = 1
dropout = 1
display_step = 1

saver = tf.train.Saver()

while step * BATCH_SIZE < training_iters:
    pass
    data, results = get_training_batch()
    #pdb.set_trace()

    sess.run(optimizer, feed_dict={x: data,  y: results, keep_prob: dropout})
    if step % display_step == 0:
        acc = sess.run(accuracy, feed_dict={x: data, y: results, keep_prob: 1.})
        loss = sess.run(cost, feed_dict={x: data, y: results, keep_prob: 1.})
        print "Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " +  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)


    step += 1

save_path = saver.save(sess, "model.ckpt")
print("Model saved in file: %s" % save_path)

print "Optimization Finished!"

data, results = get_testing_batch()
print "Testing Accuracy:",\
    sess.run(accuracy,\
    feed_dict={x: data , \
    y: results ,\
    keep_prob: 1.})

