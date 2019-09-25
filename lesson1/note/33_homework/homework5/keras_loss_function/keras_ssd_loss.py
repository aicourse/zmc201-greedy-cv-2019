'''
SSD的损失函数的 Keras 实现, 只支持TensorFlow.
'''

import tensorflow as tf

class SSDLoss:
    '''
    SSD损失函数, 参考 https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): 参与损失函数值计算的背景和真实目标的个数比例的最大值.
                真实的训练数据是没有人工标准的背景, 但是我们的`y_true`中包含了被认为是背景的Anchor.
                `y_true` 中被认为是背景的框的数量远远大于人工标注的正样本的数量. 需要做一些筛选.
                默认取值为 3.
            n_neg_min (int, optional): 每一批图像中最为负样本的背景的边界框的数量的最小值. 这个值
                确保每一次迭代梯度都有足够的负样本参与, 即便正样本的数量很小, 或者是0, 也想要使用负样
                本进行训练.
            alpha (float, optional): 用于平衡定位误差在总的误差计算中占的比重. 默认值为0.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        计算 smooth L1 loss.

        Arguments:
            y_true (nD tensor): 包含人工标准的样本的值, 形状为 `(batch_size, #boxes, 4)`. 最后维度
                包含如下四个坐标值 `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): 包含预测的数据, 和 `y_true` 的形状一样.

        Returns:
            返回smooth L1 loss, 2维 tensor, 形状 (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        计算 softmax log loss.

        Arguments:
            y_true (nD tensor): 人工标注的值, 形状 (batch_size, #boxes, #classes)
            y_pred (nD tensor): 预测的值, 与 `y_true` 形状一样.

        Returns:
            返回 softmax log loss, 2 维 tensor, 形状为 (batch, n_boxes_total).
        '''
        # 确保 `y_pred` 不包含为0的值
        y_pred = tf.maximum(y_pred, 1e-15)
        # 计算 log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        计算 SSD 模型的 loss

        Arguments:
            y_true (array): Numpy 数组, 形状 `(batch_size, #boxes, #classes + 12)`,
                其中 `#boxes` 为模型为每一幅图预测的边界框的总数. 最后的维度包含
                `[one-hot 编码的类别标签, 人工标准的边界框的 4 个坐标的偏置, 8 个随意的值]`
                其中类别标签包括背景类别的标签. 最后的8个值在此函数中没有使用, 它们的存在只是为了
                使得 `y_true` 和 `y_pred` 的形状一样.最后维度里面的最后4个值为 Anchor 的坐标, 
                在预测的时候需要使用. 如果希望将某个边界框不计入损失函数的计算, 需要将#classes对应
                的one-hot编码的值都设为0.
            y_pred (Keras tensor): 模型预测的输出. 形状和 `y_true` 一样 `(batch_size, #boxes, #classes + 12)`.
                最后一个维度包含如下格式的值
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

        Returns:
            实数值. 为定位和分类误差的值.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] # 输出类型: tf.int32
        n_boxes = tf.shape(y_pred)[1] # 输出类型: tf.int32, `n_boxes` 表达每个图像对应的所有边界框的个数, 不是特征图的每个位置对应的边界框的数量

        # 1: 为每一个边界框计算分类和定位loss 

        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # 输出形状: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # 输出形状: (batch_size, n_boxes)

        # 2: 计算正负样本的分类 losses 

        # 创建正样本, 负样本的 mask.
        negatives = y_true[:,:,0] # 形状 (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1)) # 形状 (batch_size, n_boxes)

        # 计算 y_true 中整个 batch 中正样本 (类别标签 1 到 n) 的数量.
        n_positive = tf.reduce_sum(positives)

        # 不考虑负样本的情况, 计算每一幅图图的正样本的 losses
        # (Keras 计算 loss 的时候只计算每一幅图的损失, 而不是整个 batch 的损失之和, 所以我们需要求和).
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 计算负样本的分类 loss.

        # 计算所有负样本的 loss.
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
        
        # 计算我们需要处理的负样本的数量. 最多保留 `self.neg_pos_ratio` 乘以 `y_true` 中正样本的数量, 但是至少 `self.n_neg_min` 个.
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # (1) 完全没有负样本
        # (2) 所有负样本的分类 loss 为 0, 返回 0 作为 `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss.
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False) # We don't need them sorted.
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: Compute the total loss.

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
