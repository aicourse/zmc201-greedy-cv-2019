# 
# 训练一个 SSD 网络用于识别车载摄像头捕捉的图像中的目标
# 

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

# ## 1. 设置模型参数
#
# 这些参数同时需要在 `build_model()` 和 `SSDInputEncoder` 的构造函数中使用
# 
# 
# * 设置图像的高度, 宽度, 和色彩通道数量
# * 设置正样本类别个数 (不包括背景).
# * `mode` 用于 `build_model()` 函数, 决定是否使用 `DecodeDetections` 层作为最后一层. Mode 的值为 'training' 时,
# * 网络输出原始预测值. 当值为 'inference' 或者 'inference_fast' 时, 原始的输出被转为了绝对坐标, 使用了概率筛选, 使用
# * 了 non-maximum suppression, 以及 top-k 筛选. 'inference' 使用了原始Caffe实现的算法, 'inference_fast' 使用了
# * 更快, 但是相对不太精确的算法 
# * 本例子只有 4 个预测层, 但是需要 5 个 scaling factors. 因为最后一个 scaling factor 被用于计算最后一层的第二个长宽比为 1 的 Anchor 的尺寸.
# * `build_model()` 与 `SSDInputEncoder` 有两个 Anchor 的长宽比: `aspect_ratios_global` 和 `aspect_ratios_per_layer`. 
# * 你只需要设置其中一个. 如果你想使用 `aspect_ratios_global`, 设置其值为一个Python list, 这个值会被每一层使用. 
# * 如果想使用`aspect_ratios_per_layer`, 设置其值为包含 Python list 的 Python list. 每一个 list 包含每一层的 长宽比.
# * 如果 `two_boxes_for_ar1 == True`, 那么预测层为为长宽比为一的情况预测两个边界框, 一个大一些, 一个小一些
# * 如果 `clip_boxes == True`, 那么 Anchor 会被剪切到图像边界内, 建议设置为 False.
# * 在训练的时候, 边界框相对于 Anchor 的偏置需要除以 variances, 设置为 1.0, 相当于没有使用. 使用小于 1.0 的值, 相当于使得这个维度的分布值变大. 
# * `normalize_coords` 为 True 的时候, 会将所有绝对坐标值转换为相对于图像宽度和高度的值. 这个设置不影响最终的输出.

img_height = 300 # 图像的高度
img_width = 480 # 图像的宽度
img_channels = 3 # 图像的通道数
intensity_mean = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
intensity_range = 127.5 # 用于图像归一化, 将像素值转为 `[-1,1]`
n_classes = 5 # 正样本的类别 (不包括背景)
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # Anchor 的 scaling factors. 如果设置了这个值, 那么 `min_scale` 和 `max_scale` 会被忽略
aspect_ratios = [0.5, 1.0, 2.0] # 每一个 Anchor 的长宽比
two_boxes_for_ar1 = True # 是否产生两个为长宽比为 1 的 Anchor
steps = None # 可以手动设置 Anchor 的步长, 不建议使用
offsets = None # 可以手动设置左上角 Anchor 的偏置, 不建议使用
clip_boxes = False # 是否将 Anchor 剪切到图像边界范围内 
variances = [1.0, 1.0, 1.0, 1.0] # 可以将目标的坐标 scale 的参数, 建议保留 1.0
normalize_coords = True # 是否使用相对于图像尺寸的相对坐标


# ## 2. 创建模型
# 
# 你需要使用 2.1 创建一个新的模型, 或者使用 2.2 加载一个已经训练过的模型, 不要都使用
#
# ### 2.1 创建模型
# 
# 
# 如下代码完成 3 个任务:
# 1. 调用函数 `build_model()` 创建模型.
# 2. 可选择的加载模型权值.
# 3. 编译模型, 设置优化器 (Adam) 和损失函数 (SSDLoss) 
# 
# 1: 加载模型

K.clear_session() # 从内存中清理曾经加载的模型.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# 2: 可选: 加载曾经的模型

#model.load_weights('./ssd_weights.h5', by_name=True)

# 3: 设置优化器和损失函数

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
# loss generator
# ### 2.2 加载一个训练好的模型
#
# 这里假设你想加载的模型是使用 'training' mode 训练的. 如果你想加载的模型是使用 'inference' 或者
# 'inference_fast' mode 训练的, 你需要在`custom_objects`添加 `DecodeDetections` 或者
# `DecodeDetectionsFast` 层.
#
# TODO: 设置要加载的模型的路径.
##model_path = 'ssd.h5'

# 创建 SSDLoss 对象
##ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

## K.clear_session() # 从内存中清理曾经加载的模型.

##model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
##                                               'compute_loss': ssd_loss.compute_loss})

#
# ## 3. 设置训练需要的 data generators
#
#%%
# 1: 初始化两个 `DataGenerator` 对象: 一个用于训练, 一个用于validation

# 可选: 如果你的电脑内存够大, 可以将所有的图像加载到内存中, 设置 `load_images_into_memory` = `True`.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: 加载数据

# TODO: 设置数据集地址.

# Images
images_dir = './driving_datasets/'

# Ground truth
train_labels_filename = './driving_datasets/labels_train.csv'
val_labels_filename   = './driving_datasets/labels_val.csv'

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # CSV 文件前 6 列的值
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')

# 得到训练和validation数据集的数据的量.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("训练集的图像数量:\t{:>6}".format(train_dataset_size))
print("validation集的图像数量\t{:>6}".format(val_dataset_size))


# 3: 设置 batch_size.

batch_size = 16

# 4: 定义图像增强

data_augmentation_chain = DataAugmentationConstantInputSize(random_brightness=(-48, 48, 0.5),
                                                            random_contrast=(0.5, 1.8, 0.5),
                                                            random_saturation=(0.5, 1.8, 0.5),
                                                            random_hue=(18, 0.5),
                                                            random_flip=0.5,
                                                            random_translate=((0.03,0.5), (0.03,0.5), 0.5),
                                                            random_scale=(0.5, 2.0, 0.5),
                                                            n_trials_max=3,
                                                            clip_boxes=True,
                                                            overlap_criterion='area',
                                                            bounds_box_filter=(0.3, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=(0,0,0))

# 5: 创建 encoder, 用于将人工标签转换为 SSD 损失函数需要的格式

# encoder 需要知道模型预测层输出的特征图的尺寸,用于产生 Anchor.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

# 6: 创建用于 Keras的 `fit_generator()` 的 generator.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[data_augmentation_chain],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# ## 4. 训练模型
# 
# 
# 设置 Keras callbacks. 一个用于 early stopping, 一个用于在训练看起来没有进展的情况下降低学习率,
# 一个用于保存当前最佳模型, 一个用于将训练过程的值写入 CSV 文件.
#
model_checkpoint = ModelCheckpoint(filepath='ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd7_training_log.csv',
                       separator=',',
                       append=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]

# TODO: 设置 epochs 
initial_epoch   = 0
final_epoch     = 30
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)


# 画 training and validation loss 的走势图:
plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24})
plt.show()


# ### 5. 做预测
# 
# 这里我们使用已经设置好的 validation generator 做预测
# 

# 1: 初始化 generator

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'processed_labels',
                                                  'filenames'},
                                         keep_images_without_gt=False)



# 2: 获取预测所需输入

batch_images, batch_labels, batch_filenames = next(predict_generator)

i = 0 # 图像位置

print("图像名:", batch_filenames[i])
print()
print("人工标注的值:\n")
print(batch_labels[i])



# 3: 作预测

y_pred = model.predict(batch_images)

# 4: 解码 `y_pred`
# 如果我们训练是设置的是 'inference' 或者 'inference_fast' mode, 那么模型的最后一层为 `DecodeDetections` 层,
# `y_pred` 就无需解码了. 但是我们选择了 'training' mode, 模型的原始输出需要解码. 这就是 `decode_detections()` 
# 这个函数的功能. 这个函数的功能和 `DecodeDetections` 层做的事情一样, 只是使用 Numpy 而不是 TensorFlow 实现.
# (Nunpy 只能使用CPU, 而不是GPU).
# 

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.45,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("预测值:\n")
print('   类别   概率 xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])


# 最后, 我们可以将预测的边界框画在图像上. 每一个预测的边界框都有类别名称和概率显示在边上. 
# 人工标准的边界框也使用绿色的框画出来, 便于比较.

 
# 5: 在图像上画边界框 

plt.figure(figsize=(20,12))
plt.imshow(batch_images[i])

current_axis = plt.gca()

colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # 设置边界框的颜色
classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # 类别的名称

# 画人工标注的边界框
for box in batch_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
    #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

# 画预测的边界框
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})


