import pdb
import cv2
import glob
import os

SQURE_SIZE = 64

PATH = '/home/okravchyshyn/test_cars'
PATH_TO_TRAIN = '/home/okravchyshyn/train_images'
PATH_TO_TEST = '/home/okravchyshyn/test_images'
PATH_TO_CONTOUR_IMGS = '/media/sf_CoherenSE_Files/groundtruth_GRAZ_02_900_images/carsgraz*'
PATH_TO_IMG_FILES = '/media/sf_CoherenSE_Files/cars'

#pdb.set_trace()

counter = 0

def extract_car_img(contour_fname, img_fname):
    pass
    global counter

    print contour_fname, " ", img_fname


    if os.path.isfile(img_fname) == False:
         return
    #file_mask = 'carsgraz_023_gt.jpg'
    img_car = cv2.imread(img_fname)

    img_contours = cv2.imread(contour_fname ,0)
    ret,thresh = cv2.threshold(img_contours, 127,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #if len(contours) > 1:
    #    pdb.set_trace()

    for contour in contours:
        counter = counter + 1
        x, y, w, h = cv2.boundingRect(contour)
        if w > h :
            zero_or_one = 0
            if (w - h) % 2 == 1:
                zero_or_one = 1
            delta = (w - h) / 2
            if y > delta :
                y = y - delta
            else:
                y = 0
            h = h + 2 * delta + zero_or_one
        elif h > w :
            zero_or_one = 0
            if (h - w) % 2 == 1:
                zero_or_one = 1
            delta = (h - w) / 2
            if x > delta :
                x = x - delta
            else:
                x = 0
            w = w + 2 * delta + zero_or_one

        #file_car = 'carsgraz_023.bmp'
        #img_car = cv2.imread(PATH + '/' + file_car)
        img_sel = img_car[y:y + h,x:x+w,:]
        img_final = cv2.resize(img_sel,(SQURE_SIZE, SQURE_SIZE), interpolation = cv2.INTER_AREA)
        final_fname = 'car%03d.jpg' % (counter)
        print final_fname
        cv2.imwrite( PATH_TO_TRAIN  + '/' + final_fname, img_final)



#area = cv2.contourArea(contours)

pass

def process_directory_with_contour_files():
    contour_files = glob.glob(PATH_TO_CONTOUR_IMGS)
    for f in contour_files:
        pass
        full_fname = os.path.basename(f)
        fname, ext = os.path.splitext(full_fname)
        img_basename = fname[:-3] + '.' + 'bmp'
        img_fname = PATH_TO_IMG_FILES + '/' + img_basename
        #pdb.set_trace()
        extract_car_img(f, img_fname)


process_directory_with_contour_files()
#f = '/media/sf_CoherenSE_Files/groundtruth_GRAZ_02_900_images/carsgraz_29_gt.jpg'
#img_fname = '/media/sf_CoherenSE_Files/cars/carsgraz_29.bmp'
#pdb.set_trace()
#extract_car_img(f, img_fname)

