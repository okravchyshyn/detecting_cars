import pdb
import cv2
import glob
import os

SQURE_SIZE = 64

PATH_TO_TEST = '/home/okravchyshyn/test_images'
PATH_TO_NOCAR_IMG = '/media/sf_CoherenSE_Files/none/*.bmp'
PATH_TO_TRAIN = '/home/okravchyshyn/train_nocar_images'

#pdb.set_trace()

counter = 0

def extract_nocar_img(nocar_fname):
    pass
    global counter

    print nocar_fname


    if os.path.isfile(nocar_fname) == False:
         return
    
    counter = counter + 1
    #file_mask = 'carsgraz_023_gt.jpg'
    img_nocar = cv2.imread(nocar_fname)

    h, w, _ = img_nocar.shape
    if w > h:
       w = h
    else:
       h = w

    img_sel = img_nocar[0:h,0:w,:]
    img_final = cv2.resize(img_sel,(SQURE_SIZE, SQURE_SIZE), interpolation = cv2.INTER_AREA)
    final_fname = 'nocar%03d.jpg' % (counter)
    print final_fname
    cv2.imwrite( PATH_TO_TRAIN  + '/' + final_fname, img_final)



#area = cv2.contourArea(contours)

pass

def process_directory_with_nocar_files():
    nocar_files = glob.glob(PATH_TO_NOCAR_IMG)
    for f in nocar_files:
        #pdb.set_trace()
        extract_nocar_img(f)


process_directory_with_nocar_files()
#f = '/media/sf_CoherenSE_Files/groundtruth_GRAZ_02_900_images/carsgraz_29_gt.jpg'
#img_fname = '/media/sf_CoherenSE_Files/cars/carsgraz_29.bmp'
#pdb.set_trace()
#extract_car_img(f, img_fname)

