from cv2 import *


def find_appart(start, width, black, white, black_max, white_max, arg):
    c = 0.95  # change in here
    # end = start + 1
    for m in range(start+1, width-1):
        if (black[m] if arg else white[m]) > (c * black_max if arg else c * white_max):
            end = m
            break
    return end


def image_deal(filepath):
    img = imread(filepath)
    img_gray = cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_thre = img_gray
    threshold(img_gray, 100, 255, THRESH_BINARY_INV, img_thre)
    imshow('thre', img_thre)
    waitKey(0)

    imwrite('./image/thre_res.png', img_thre)

    white = []
    black = []
    height = img_thre.shape[0]
    width = img_thre.shape[1]

    white_max, black_max = 0, 0

    for i in range(width):
        s, t = 0, 0
        for j in range(width):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)
        print(s, " ", t)

    arg = False
    if black_max > white_max:
        arg = True

    n = 1
    start = 1
    end = 2
    while n < width-2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            start = n
            end = find_appart(start, width, black, white, black_max, white_max, arg)
            n = end
            if end - start > 5:
                cj = img_thre[1:height, start:end]
                imshow('des', cj)
                waitKey(0)


image_deal('./image/src.jpeg')
