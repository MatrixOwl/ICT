import h5py
import json

import cv2
import numpy as np


def dex(img_data):
    global x1, y1, x2, y2
    fak = 0
    row, col = img_data.shape
    for r in range(row):
        for c in range(col):
            if int(img_data[r, c]) > fak:
                y1 = r
    for c in range(col):
        for r in range(row):
            if int(img_data[r, c]) > fak:
                x1 = c
    for c in range(col-1, 0, -1):
        for r in range(row):
            if int(img_data[r, c]) > fak:
                x2 = c
    for r in range(row-1, 0, -1):
        for c in range(col):
            if int(img_data[r, c]) > fak:
                y2 = r

    return x1, y1, x2, y2


all_data = []
number_dis = 3

h5file = h5py.File('train_img.hdf5', 'w')
h5data = h5file.create_dataset("train", (number_dis, ))

x_new = np.load('x_train.npy').reshape(-1, 28, 28, 1).astype('float32')
y_new = np.load('y_train.npy').astype('uint8')

for d_p in range(number_dis):
    num_pic = np.random.randint(6, 15)  # 6, 15

    image_back = np.zeros((300, 300, 1), np.uint8)   # 300, 300
    image_back.fill(255)
    dis_name = "./save/" + str(d_p) + ".jpg"

    dis_name_match = str(d_p)

    dist = []

    for i in range(num_pic):

        d = np.random.randint(0, 3000, 1)

        image_part_d = x_new[d]
        n_ar = y_new[d]
        n = int(n_ar[0])

        image_part = np.mat(image_part_d)

        size1 = np.random.randint(25, 28)
        size2 = size1

        image_part_in = cv2.resize(image_part, (size1, size2), interpolation=cv2.INTER_CUBIC)

        (h, w) = image_part_in.shape[:2]
        center = (w/2, h/2)
        pot = np.random.randint(-20, 20)
        M = cv2.getRotationMatrix2D(center, pot, 1.0)
        image_part_in_new = cv2.warpAffine(image_part_in, M, (w, h))

        x = np.random.randint(0, 270)  # 0, 270
        y = np.random.randint(0, 270)  # 0, 270
        if len(dist):
            for xi, yi in dist:
                if -30 < x - xi < 30 or -30 < y - yi < 30:
                    while -30 < x - xi < 30 or -30 < y - yi < 30:
                        x = np.random.randint(0, 270)  # 0, 270
                        y = np.random.randint(0, 270)  # 0, 270
            dist.append([x, y])
        else:
            dist.append([x, y])

        roi = image_back[x:x+size1, y:y+size2]

        rows, cols = image_part_in_new.shape

        p1, q1, p2, q2 = dex(image_part_in_new)
        # print(p1, ' ', q1, ' ', p2, ' ', q2)

        for k in range(rows):
            for j in range(cols):
                if int(image_part_in_new[k, j]) >= 80:  # 45  #inage_part
                    roi[k, j] = 0
                else:
                    roi[k, j] = 255
        image_back[x:x + size1, y:y + size1] = roi

        # cv2.rectangle(image_back, (y, x), (y+size2, x+size1), 0, 1)
        cv2.rectangle(image_back, (q1+y, p1+x), (q2+y, p2+x), 0, 1)
        cv2.putText(image_back, str(n),  (y-2, x-2), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 1)

        print('path: ', dis_name, 'number: ', n, 'x1: ', x, 'y1: ', y, 'x2: ', x + size1, 'y2: ', y + size2)
        json_data = {'match_label': dis_name_match, 'number': n, 'x1': x, 'y1': y, 'x2': x + size1, 'y2': y + size2}
        all_data.append(json_data)
    """
    ret, re_img = cv2.threshold(image_back.copy(), 127, 255, cv2.THRESH_BINARY_INV)
    re_img = cv2.medianBlur(re_img, 3)
    contours, hierarchy = cv2.findContours(re_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(image_back, (x, y), (x + w, y + h), 0, 2)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_ = np.int64(box)
        cv2.drawContours(image_back, [box_], 0, 0, 1)
    """
    cv2.imwrite(dis_name, image_back)
    # np.save("train_image.npy", image_back)

    h5file[str(d_p)] = image_back
    print("done")

with open('train.json', "wb") as f:
    for data in all_data:
        f.write(json.dumps(data).encode("utf-8"))
