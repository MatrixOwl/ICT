import pprint
import time
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.metrics import average_precision_score
from keras.models import Model
from frcnn.function import *

C = Config()
base_path = ''

train_path = 'train_info_c.txt'  # Training data (annotation file)

test_path = 'validation_info_c.txt'  # Test data (annotation file)

test_base_path = base_path + 'validation_image'  # Directory to save the test images

num_rois = 2  # Number of RoIs to process at once.

# Augmentation flag
horizontal_flips = True  # Augment with horizontal flips in training.
vertical_flips = True  # Augment with vertical flips in training.
rot_90 = True  # Augment with 90 degree rotations in training.

output_weight_path = os.path.join(base_path, 'model/model_frcnn.hdf5')

record_path = os.path.join(base_path, 'model/record.csv')
# Record data (used to save the losses, classification accuracy and mean average precision)

base_weight_path = os.path.join(base_path, 'model/weights_tf_dim_ordering_tf_kernels.h5')

config_output_filename = os.path.join(base_path, 'model_config.pickle')

C.record_path = record_path
C.model_path = output_weight_path
C.num_rois = num_rois

C.base_net_weights = base_weight_path

st = time.time()
train_imgs, classes_count, class_mapping = get_data(train_path)
# print()
# print('Spend %0.2f mins to load the data' % ((time.time() - st) / 60))

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)
# e.g.
#    classes_count: {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
#    class_mapping: {'Person': 0, 'Car': 1, 'Mobile phone': 2, 'bg': 3}
C.class_mapping = class_mapping

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print(class_mapping)

# Save the configuration
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

# Shuffle the images with seed
random.seed(1)
random.shuffle(train_imgs)

print('Num train samples (images) {}'.format(len(train_imgs)))

# Get train data generator which generate X, Y, image_data
data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length, mode='train')

X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    # img[:, :, 0] -= C.img_channel_mean[0]
    # img[:, :, 1] -= C.img_channel_mean[1]
    # img[:, :, 2] -= C.img_channel_mean[2]
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    # img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    ratio = 1.0
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


num_features = 128

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)

classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

# Switch key value for class mapping
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

test_imgs = os.listdir(test_base_path)

imgs_path = []
for i in range(12):
    idx = np.random.randint(len(test_imgs))
    imgs_path.append(test_imgs[idx])

all_imgs = []

classes = {}

X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)
test_imgs, _, _ = get_data(test_path)
data_gen_test = get_anchor_gt(test_imgs, C, get_img_output_length, mode='test')

# If the box classification value is less than this, we ignore this box
bbox_threshold = 0.7

for idx, img_name in enumerate(imgs_path):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(test_base_path, img_name)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)

    X = np.transpose(X, (0, 2, 3, 1))
    # X, _, _, _, _ = next(data_gen_train)
    # get output layer Y1, Y2 from the RPN and the feature maps F
    # Y1: y_rpn_cls
    # Y2: y_rpn_regr
    [Y1, Y2, F] = model_rpn.predict(X)
    # #[Y1, Y2] = model_rpn.predict(X)
    # Get bboxes by applying NMS
    # R.shape = (300, 4)
    R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
        # #[P_cls, P_regr] = model_classifier.predict([X, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            if jk > 5:
                break
            (x1, y1, x2, y2) = new_boxes[jk, :]

            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (255, 255, 255), 1)
            # int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])
            textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
            all_dets.append((key, 100 * new_probs[jk]))

            (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
            textOrg = (real_x1, real_y1 - 0)
            # baseLine=5
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] - 5, textOrg[1] - retval[1] - 5), (200, 0, 0), 1)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] - 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, (textOrg[0] - 5, textOrg[1] + baseLine - 5), cv2.FONT_HERSHEY_DUPLEX, 0.3,
                        (0, 0, 0), 1)

    print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    plt.figure(figsize=(10, 10))
    # plt.grid('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

for i in range(20):
    coy = 33
    path = 'validation_image/validation_image_' + str(coy+i) + '.jpg'
    X = cv2.imread(path).reshape(1, 200, 200, 3) / 255
    P_rpn = model_rpn.predict_on_batch(X)
    R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=100)
    # img = debug_img.copy()
    img = plt.imread(path)
    for i in range(R.shape[0]):
        cv2.rectangle(img, (R[i][0] * 8, R[i][1] * 8), (R[i][2] * 8, R[i][3] * 8), (255, 0, 0), 1)
    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.imshow(img)
    # plt.show()


def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou_map = iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            if iou_map >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))
    # 漏检的计入
    for gt_box in gt:
        if not gt_box['bbox_matched']:  # and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    # import pdb
    # pdb.set_trace()
    return T, P


def format_img_map(img, C):
    """Format image for mAP. Resize original image to C.im_size (300 in here)

    Args:
        img: cv2 image
        C: config

    Returns:
        img: Scaled and normalized image with expanding dimension
        fx: ratio for width scaling
        fy: ratio for height scaling
    """

    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # Change image channel from BGR to RGB
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    # img[:, :, 0] -= C.img_channel_mean[0]
    # img[:, :, 1] -= C.img_channel_mean[1]
    # img[:, :, 2] -= C.img_channel_mean[2]
    img /= 255.0
    # Change img shape from (height, width, channel) to (channel, height, width)
    img = np.transpose(img, (2, 0, 1))
    # Expand one dimension at axis 0
    # img shape becames (1, channel, height, width)
    img = np.expand_dims(img, axis=0)
    return img, fx, fy


print(class_mapping)

img = debug_img.copy()

# This might takes a while to parser the data
test_imgs, _, _ = get_data(test_path)
(img <= 255).all()
X, _, _ = format_img_map(img, C)
(X <= 1).all()

T = {}
P = {}
mAPs = []
bbox_threshold = 0.7
for idx, img_data in enumerate(test_imgs):
    print('{}/{}'.format(idx, len(test_imgs)))
    st = time.time()
    filepath = img_data['filepath']

    img = cv2.imread(filepath)

    X, fx, fy = format_img_map(img, C)

    # Change X (img) shape from (1, channel, height, width) to (1, height, width, channel)
    X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate all classes' bboxes coordinates on resized image (300, 400)
        # Drop 'bg' classes bboxes
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            # Get class name
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    print('bboxes.shape[0]: %d' % (P_cls.shape[1]))
    for key in bboxes:
        bbox = np.array(bboxes[key])

        # Apply non-max-suppression on final bboxes to get the output bounding boxe
        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)

        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            all_dets.append(det)

            # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 1)

            textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))

            (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
            textOrg = (real_x1, real_y1 - 0)
            # baseLine=5
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] - 5, textOrg[1] - retval[1] - 5), (200, 0, 0), 1)
            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          (textOrg[0] + retval[0] - 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, (textOrg[0] - 5, textOrg[1] + baseLine - 5), cv2.FONT_HERSHEY_DUPLEX, 0.3,
                        (0, 0, 0), 1)
    cv2.imwrite('predict_image/img_%d.jpg' % idx, img)
    # plt.figure(figsize=(10,10))
    # plt.grid('off')
    # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #######
    print('Elapsed time = {}'.format(time.time() - st))
    # a = all_dets
    t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
    for key in t.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])
    all_aps = []
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
    print('mAP = {}'.format(np.mean(np.array(all_aps))))
    mAPs.append(np.mean(np.array(all_aps)))
    # print(T)
    # print(P)

print()
print('mean average precision:', np.mean(np.array(mAPs)))

record_df = pd.read_csv(record_path)
mAP = [mAP for mAP in mAPs if str(mAP) != 'nan']
mean_average_prec = round(np.mean(np.array(mAP)), 3)
print('After training %dk batches, the mean average precision is %0.3f' % (len(record_df), mean_average_prec))

# record_df.loc[len(record_df)-1, 'mAP'] = mean_average_prec
# record_df.to_csv(C.record_path, index=0)
# print('Save mAP to {}'.format(C.record_path))
