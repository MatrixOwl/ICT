import cv2
global img
global point1, point2


def do_image():
    global img
    img = cv2.imread('./images/src.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image')
    # cv2.setMouseCallback('image', on_mouse)
    # cv2.imshow('image', img)
    resize_img = cv2.resize(img, (28, 28))
    ret, thresh_img = cv2.threshold(resize_img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('result', thresh_img)
    cv2.imwrite('./images/text.png', thresh_img)
    cv2.waitKey(0)


# main()