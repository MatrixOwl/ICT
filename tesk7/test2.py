from matplotlib import pyplot as plt
import pandas as pd
from frcnn.function import *


record_path = './model/record.csv'
record_df = pd.read_csv(record_path)
r_epochs = len(record_df)
print(r_epochs)

plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')  # r_epochs + 1
plt.title('mean_overlapping_bboxes')
plt.subplot(1, 2, 2)
plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
plt.title('class_acc')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
plt.title('loss_rpn_cls')
plt.subplot(1, 2, 2)
plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
plt.title('loss_rpn_regr')
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
plt.title('loss_class_cls')
plt.subplot(1, 2, 2)
plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
plt.title('loss_class_regr')
plt.show()

plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
plt.title('total_loss')
plt.show()
