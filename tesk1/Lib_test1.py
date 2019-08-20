import numpy as np
import PIL.Image as image
from PIL import ImageFilter
from sklearn.cluster import KMeans


def log_pic(pic_m, v):
    g = np.log2(1 + v * pic_m) / np.log2(v + 1)
    img_m = (g * 255)
    return img_m


def loadData(filePath):
    f = open(filePath, 'rb')
    data = []
    img = image.open(f)

    img = img.filter(ImageFilter.RankFilter(3, 8))

    m, n = img.size
    print(m, n)
    for i in range(m):
        for j in range(n):
            # RGB 0-1
            x, y, z = img.getpixel((i, j))
            # color in data
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return log_pic(np.mat(data), 10), m, n


def split_img(filepath, p):
    f = open(filepath, 'rb')
    img = image.open(f)
    m, n = img.size

    for i in range(m):
        for j in range(n):
            if p.getpixel((i, j)) == 1:
                img.putpixel((i, j), 0)

    return img
# start to work


def work(file):
    imgData, row, col = loadData(file)

    km = KMeans(n_clusters=5)

    label = km.fit_predict(imgData)
    label = label.reshape([row, col])

    pic_new = image.new("L", (row, col))

    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), int(256/(label[i][j]+1)))

    pic_new = pic_new.filter(ImageFilter.ModeFilter(7))
    back = pic_new.getpixel((50, 170))
    pic_new = pic_new.point(lambda x: 0 if x == back else 1)

    pic_new = split_img(file, pic_new)
    # return pic_new
    pic_new.save("./Image_result1/7_2_result.jpg", "JPEG")


work('./Image_data/7.jpeg')
