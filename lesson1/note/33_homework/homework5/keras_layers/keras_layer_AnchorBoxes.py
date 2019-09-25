'''
用于产生 Anchor 的 Keras 层
'''

import numpy as np
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer

from bounding_box_utils.bounding_box_utils import convert_coordinates

class AnchorBoxes(Layer):
    '''
    用于产生 Anchor 和 variance 的 Keras 层.

    为输入的 tensor 的每一个空间位置, 产生一组具有不同长宽比的 anchor. 每一个空间位置产生的
    anchor 的数量取决于 `aspect_ratios` 和 `two_boxes_for_ar1`, 默认情况下为 4. 返回
    anchor 的坐标格式为 `(xmin, xmax, ymin, ymax)`.

    逻辑与 `ssd_box_encode_decode_utils.py` 相似.

    有了这一层, 使用训练好的 model 做预测的时候不需要调用额外的函数用于产生 Anchor. 因为 model 的预测
    值是一个相对值, 必须要知道 Anchor 的坐标, 才能够算出边界框的绝对值.

    输入 tensor 的形状:
        4 维, 如果 `dim_ordering = 'th'`, 则 `(batch, channels, height, width)` 
              如果 `dim_ordering = 'tf'`, 则 `(batch, height, width, channels)`

    输出 tensor 的形状:
        5 维, `(batch, height, width, n_boxes, 8)`. 最后的 1 个维度上的值分别为每个 Anchor 的4
        个坐标和 4 个variance.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 coords='centroids',
                 normalize_coords=False,
                 **kwargs):
        '''
        所有的参数的值应该和边界框编码过程的值保持一致.

        Arguments:
            img_height (int): 输入图像的高度.
            img_width (int): 输入图像的宽度.
            this_scale (float): [0, 1]之间的浮点数, 表达了 anchor 的尺寸相对于输入图像较短的那一个边的比例.
            next_scale (float): [0, 1]之间的浮点数, 下一层的 anchor 的尺寸相对于输入图像较短的那一个边的比例. 只有当
                `self.two_boxes_for_ar1 == True` 的时候才有意义.
            aspect_ratios (list, optional): 产生的 anchor 的长宽比.
            two_boxes_for_ar1 (bool, optional): 只有当 `aspect_ratios` 包含 1 才有意义.
                当值为 `True` 时, 对应长宽比为 1 的情况, 会产生两个 Anchor. 第一个 Anchor 的尺寸使用当前层的 scale. 
                第二个 Anchor 的尺寸使用当前与较大的下一层的 scale 的几何平均值.
            clip_boxes (bool, optional): 当值为 `True`, 将 Anchor 的坐标限定在图像范围内.
            variances (list, optional): 4 个大于 0 的浮点数. Anchor 每一个坐标都除以这个值. 没有特别物理意义, 只是为了数值稳定. 取值为 1, 相当于没有.
            coords (str, optional): 训练模型内部使用的边界框的坐标格式 (不是输入人工标签的坐标格式). 可以为 'centroids' `(cx, cy, w, h)` (边界的中心坐标, 宽, 高),
                'corners' `(xmin, ymin, xmax,  ymax)`, 或者 'minmax' for the format `(xmin, xmax, ymin, ymax)`.
            normalize_coords (bool, optional): 如果模型使用的相对值, 设置为 `True`. 即 model 预测的边界框坐标值在 [0,1] 之间而非绝对像素值坐标.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("这一层只支持 tensorflow, 但是你使用了 {} backend.".format(K.backend()))

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        # 计算每一个特征图像素对应的 Anchor 的数量
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        根据输入 tensor 的形状, 计算 Anchor tensor.

        此处的逻辑和 `ssd_box_encode_decode_utils.py` 一样.

        请注意这些 tensor 并不参与相互误差传递的计算优化过程. 其值为常数. 因此, 所有的逻辑使用 Numpy 计算,只是在最后才
        转换为 Keras tensor 输出.

        Arguments:
            x (tensor): 4 维 tensor, 当 `dim_ordering = 'th'`, 形状为 `(batch, channels, height, width)`,
            当 `dim_ordering = 'tf'`, 形状为 `(batch, height, width, channels)`.
        '''

        # 计算每一个长宽比下, Anchor 的宽和高
        # 输入图像的较短的边长, 与`scale` 和 `aspect_ratios` 一起用于计算
        size = min(self.img_height, self.img_width)
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # 对应长宽比为 1 的情况计算 anchor .
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # 使用本层和下一层的 scale 的几何平均值计算稍微大一点的 Anchor 的尺寸
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        # 获取输入 tensor 的形状
        
        batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        # 计算 Anchor 的中心点位置, 对不同的长宽比, 中心点位置是同样的.

        # 计算步长. 计算两个相邻 Anchor 的中心的纵横方向的距离
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # 计算偏移量. 第一个 Anchor 相对于输入图像的左上角的偏移量
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        # 有了偏移量和步长, 计算 Anchor 的中心位置
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # 为了如下的 np.tile() 做准备
        cy_grid = np.expand_dims(cy_grid, -1) # 为了如下的 np.tile() 做准备

        # 产生一个 4 维的 tensor 模版, 形状为 `(feature_map_height, feature_map_width, n_boxes, 4)`
        # 最后 4 维的值为 `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # 设置 cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # 设置 cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # 设置 w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # 设置 h

        # 坐标转换 `(cx, cy, w, h)` 到 `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # 如果 `clip_boxes` 的值为 ‘True’, 剪切坐标, 使其的值在图像边界以内
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # 如果 `normalize_coords` 的值为 ‘True’, 归一化坐标, 使得其值在 [0,1] 之间
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        if self.coords == 'centroids':
            # 转换 `(xmin, ymin, xmax, ymax)` 到 `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # 转换 `(xmin, ymin, xmax, ymax)` 到 `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        # 创建一个 tensor 保护 variances , 将它添加到 `boxes_tensor` 中. 这个 tensor 的形状和 `boxes_tensor` 的一样
        variances_tensor = np.zeros_like(boxes_tensor) # 形状为 `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances 
        # 拼接以后 `boxes_tensor` 的形状为 `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # 在 `boxes_tensor` 前面加一个 维度, 作为 batch size 的维度
        # 拼接以后维度为 5 维, 形状为 `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
