import cv2
import numpy as np
import glob
import pdb

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




training_samples = 0


BATCH_SIZE = 20


def load_training_data():

    global training_data
    global training_result

    global testing_data
    global testing_result

    global number_of_training_samples
    global number_of_testing_samples
    
    img_buffer = np.zeros((SQURE_SIZE, SQURE_SIZE,3), dtype = np.float32) 

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


    training_data = np.zeros((number_of_training_samples , SQURE_SIZE, SQURE_SIZE, 3), np.float32)
    training_result = np.zeros(number_of_training_samples)

    testing_data = np.zeros((number_of_testing_samples , SQURE_SIZE, SQURE_SIZE, 3), np.float32)
    testing_result = np.zeros(number_of_testing_samples )


    idx = 0

    for i in range(number_of_training_samples):

        idx_in_car_files  =  random_all_sample_idx[i]
        f = merged_files[idx_in_car_files]
        r = merged_results[idx_in_car_files]
        print idx_in_car_files, " " , f, " result = " , r 
        img = cv2.imread(f)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        img_buffer = img.astype(np.float32)/255

        training_data[i] = img_buffer
        training_result[i] = r

    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print 'Testing data'
    print ''
    pdb.set_trace()

    for j in  range(number_of_testing_samples) :
        i = j + number_of_training_samples

        idx_in_car_files  =  random_all_sample_idx[i]
        f = merged_files[idx_in_car_files]
        r = merged_results[idx_in_car_files]
        print idx_in_car_files, " " , f, " result = " , r 
        img = cv2.imread(f)
        img_buffer = img.astype(np.float32)/255

        testing_data[j] = img_buffer
        testing_result[j] = r


def get_training_batch():
    batch_data = np.zeros((BATCH_SIZE, SQURE_SIZE, SQURE_SIZE, 3), np.float32)
    batch_result = np.zeros(BATCH_SIZE)
    begin_from_idx = np.random.randint(number_of_training_samples)  

    for i in range(BATCH_SIZE):
        j = (begin_from_idx + i) % number_of_training_samples
 
        batch_data[i] = training_data[j]
        batch_result[i] = training_result[j]

    return batch_data, batch_result


def get_testing_batch():
    batch_data = np.zeros((BATCH_SIZE, SQURE_SIZE, SQURE_SIZE, 3), np.float32)
    batch_result = np.zeros(BATCH_SIZE)
    begin_from_idx = np.random.randint(number_of_testing_samples)  

    for i in range(BATCH_SIZE):
        j = (begin_from_idx + i) % number_of_testing_samples
 
        batch_data[i] = testing_data[j]
        batch_result[i] = testing_result[j]

    return batch_data, batch_result


pdb.set_trace()
load_training_data()

pdb.set_trace()
data, results = get_training_batch()

