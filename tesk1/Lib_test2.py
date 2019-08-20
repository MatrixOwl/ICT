import numpy as np
import PIL.Image as image
from PIL import ImageFilter
from sklearn.cluster import KMeans


def loadData(filePath):
    f = open(filePath, 'rb')
    data = []
    img = image.open(f)

    img = img.filter(ImageFilter.RankFilter(3, 8))

    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z, a = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0, a / 256.0])
    f.close()
    return np.mat(data), m, n


color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 255, 255), (0, 0, 0),
         (85, 185, 125), (230, 58, 125), (150, 250.55)]

imgData, row, col = loadData('./Image_data/2.png')

km = KMeans(n_clusters=10)

label = km.fit_predict(imgData)
label = label.reshape([row, col])

pic_new = image.new("RGB", (row, col))

for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), color[label[i][j]])

pic_new.save("./Image_result2/arf_result2.png", "PNG")
