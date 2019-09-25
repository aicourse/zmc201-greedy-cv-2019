'''
定制的 Keras 层, 用于解码 SSD 预测的输出. 这一个实现和 SSD 论文的标准 Caffe 实现不一致, 更加快.
'''

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

class DecodeDetectionsFast(Layer):
    '''
    定制的 Keras 层用于解码 SSD 的预测输出.

    输入形状:
        3 维, 形状 `(batch_size, n_boxes, n_classes + 12)`.

    Output shape:
        3 维, 形状 `(batch_size, top_k, 6)`.
    '''

    def __init__(self,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=200,
                 nms_max_output_size=400,
                 coords='centroids',
                 normalize_coords=True,
                 img_height=None,
                 img_width=None,
                 **kwargs):
        '''
        All default argument values follow the Caffe implementation.

        Arguments:
            confidence_thresh (float, optional): [0,1) 之间的浮点数, 能够被认为是目标的最小类别输出概率. 值越小, 接下来的 non-maximum suppression
                需要处理的候选框越多. 
            iou_threshold (float, optional): [0,1] 之间的浮点数. 所有和同一个类别具有最大的类别概率的边界框IoU大于iou_threshold的边界框会被去除.
            top_k (int, optional): 在non-maximum suppression之后, 需要保留具有最大类别概率的边界框的个数
            nms_max_output_size (int, optional): non-maximum suppression之后输出的边界框的个数的最大值.
            coords (str, optional): 模型输出的边界框的坐标的格式, 暂时只支持'centroids' 格式, 
                即 `(cx, cy, w, h)` (中心位置, 宽, 高). 
            normalize_coords (bool, optional): 如果模型的输出坐标为相对值, 大小在 [0,1] 之间, 而且你希望坐标转换为绝对值, 设置为 `True` .
                如果模型的输出坐标为相对值, 大小在 [0,1] 之间, 而且你不希望坐标转换为绝对值, 设置为 `False` .
                如果模型的输出本来就是绝对值, 不能设置为 `True`. 
                设置为 `True` 的时候需要输入 `img_height` 和 `img_width`.
            img_height (int, optional): 输入图像的高度. 只有当 `normalize_coords` 设置为 `True` 时才需要.
            img_width (int, optional): 输入图像的宽度. 只有当 `normalize_coords` 设置为 `True` 时才需要.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("当前只支持 tensorflow, 但是你使用了 {} backend.".format(K.backend()))

        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError("如果使用相对坐标, 并且想转换为绝对坐标, 那么需要传人图像大小的值. 传入的值为 `img_height == {}` and `img_width == {}`".format(img_height, img_width))

        if coords != 'centroids':
            raise ValueError("此层只支持 'centroids' 坐标格式.")

        # Keras 的层 config 需要这些值.
        self.confidence_thresh = confidence_thresh
        self.iou_threshold = iou_threshold
        self.top_k = top_k
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width
        self.coords = coords
        self.nms_max_output_size = nms_max_output_size

        # TensorFlow 需要这些常数.
        self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(self.top_k, name='top_k')
        self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
        self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
        self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

        super(DecodeDetectionsFast, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(DecodeDetectionsFast, self).build(input_shape)

    def call(self, y_pred, mask=None):
        '''
        Returns:
            3 维tensor, 形状 `(batch_size, top_k, 6)`. 第二维度包含 `top_k` 个预测值.
            最后一个维度包含如下值`[类别标签, 概率, xmin, ymin, xmax, ymax]`.
        '''

        #####################################################################################
        # 1. 将边界框坐标从预测的相对于 Anchor 的偏置, 转换为绝对值坐标
        #####################################################################################

        # 提取每一个 Anchor 对应的预测类别中概率最高的类别的标签 
        class_ids = tf.expand_dims(tf.to_float(tf.argmax(y_pred[...,:-12], axis=-1)), axis=-1)
        # 提取每一个 Anchor 位置对应的最大的概率.
        confidences = tf.reduce_max(y_pred[...,:-12], axis=-1, keep_dims=True)

        # 从相对于 Anchor 的偏置转换为相对于输入图像的偏置
        cx = y_pred[...,-12] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cy = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        w = tf.exp(y_pred[...,-10] * y_pred[...,-2]) * y_pred[...,-6] # w = exp(w_pred * variance_w) * w_anchor
        h = tf.exp(y_pred[...,-9] * y_pred[...,-1]) * y_pred[...,-5] # h = exp(h_pred * variance_h) * h_anchor

        # 将坐标格式从 'centroids' 转为 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        # 如果模型的输出边界是相对于图像尺寸的相对值, 而且我们希望转为绝对尺寸, 需要如下转换
        def normalized_coords():
            xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
            ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
            xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
            ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
            return xmin1, ymin1, xmax1, ymax1
        def non_normalized_coords():
            return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

        xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalized_coords, non_normalized_coords)

        # 将预测的 one-hot 编码的类别概率和边界框的坐标合并产生预测输出的 tensor
        y_pred = tf.concat(values=[class_ids, confidences, xmin, ymin, xmax, ymax], axis=-1)

        #####################################################################################
        # 2. 使用概率阈值筛选边界框, 进行每一个类别的 non-maximum suppression, 最后选择 top-k 个边界框.
        #####################################################################################

        batch_size = tf.shape(y_pred)[0] # 输出类型: tf.int32
        n_boxes = tf.shape(y_pred)[1]
        n_classes = y_pred.shape[2] - 4
        class_indices = tf.range(1, n_classes)

        # 创建一个函数, 筛选预测的边界框, 完成如下任务:
        # - 按照概率阈值筛选边界框
        # - non-maximum suppression (NMS)
        # - top-k 筛选
        def filter_predictions(batch_item):

            # 只保留非背景的边界框.
            positive_boxes = tf.not_equal(batch_item[...,0], 0.0)
            predictions = tf.boolean_mask(tensor=batch_item,
                                          mask=positive_boxes)

            def perform_confidence_thresholding():
                # 按照概率筛选
                threshold_met = predictions[:,1] > self.tf_confidence_thresh
                return tf.boolean_mask(tensor=predictions,
                                       mask=threshold_met)
            def no_positive_boxes():
                return tf.constant(value=0.0, shape=(1,6))

            # 如果边界框的个数大于0, 按照概率进行筛选.
            predictions_conf_thresh = tf.cond(tf.equal(tf.size(predictions), 0), no_positive_boxes, perform_confidence_thresholding)

            def perform_nms():
                scores = predictions_conf_thresh[...,1]

                # 函数 `tf.image.non_max_suppression()` 需要边界框的格式为 `(ymin, xmin, ymax, xmax)`.
                xmin = tf.expand_dims(predictions_conf_thresh[...,-4], axis=-1)
                ymin = tf.expand_dims(predictions_conf_thresh[...,-3], axis=-1)
                xmax = tf.expand_dims(predictions_conf_thresh[...,-2], axis=-1)
                ymax = tf.expand_dims(predictions_conf_thresh[...,-1], axis=-1)
                boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                              scores=scores,
                                                              max_output_size=self.tf_nms_max_output_size,
                                                              iou_threshold=self.iou_threshold,
                                                              name='non_maximum_suppresion')
                maxima = tf.gather(params=predictions_conf_thresh,
                                   indices=maxima_indices,
                                   axis=0)
                return maxima
            def no_confident_predictions():
                return tf.constant(value=0.0, shape=(1,6))

            # 进行 NMS.
            predictions_nms = tf.cond(tf.equal(tf.size(predictions_conf_thresh), 0), no_confident_predictions, perform_nms)

            # 进行 top-k 的筛选, 如果余下的边界框的数量不足 ‘top_k', 则填充到 ‘top_k'. 使得产生的tensor
            # 长度为 `self.top_k`. 
            def top_k():
                return tf.gather(params=predictions_nms,
                                 indices=tf.nn.top_k(predictions_nms[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)
            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=predictions_nms,
                                            paddings=[[0, self.tf_top_k - tf.shape(predictions_nms)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(predictions_nms)[0], self.tf_top_k), top_k, pad_and_top_k)

            return top_k_boxes

        # 对同一个 batch 的所有图像进行 `filter_predictions()` 
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return (batch_size, self.tf_top_k, 6) # Last axis: (class_ID, confidence, 4 box coordinates)

    def get_config(self):
        config = {
            'confidence_thresh': self.confidence_thresh,
            'iou_threshold': self.iou_threshold,
            'top_k': self.top_k,
            'nms_max_output_size': self.nms_max_output_size,
            'coords': self.coords,
            'normalize_coords': self.normalize_coords,
            'img_height': self.img_height,
            'img_width': self.img_width,
        }
        base_config = super(DecodeDetectionsFast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
