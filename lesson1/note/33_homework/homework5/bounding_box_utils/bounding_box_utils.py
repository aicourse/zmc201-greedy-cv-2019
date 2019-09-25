import numpy as np

def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    转换编辑的坐标格式, 支持如下格式互转:
        1) (xmin, xmax, ymin, ymax) - 'minmax' 格式
        2) (xmin, ymin, xmax, ymax) - 'corners' 格式
        2) (cx, cy, w, h) - 'centroids' 格式

    Arguments:
        tensor (array): Numpy n 维数组, 包含需要转换的原始坐标.
        start_index (int): 输入数组中保存第一个坐标的位置的下标.
        conversion (str, optional): 坐标转换的方向. 可以为 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            'corners2minmax'.
        border_pixels (str, optional): 如何处理位于边界框上的像素. 取值可以为:
            'include': 边界上的像素属于目标的一部分
            'exclude': 边界上的像素不属于目标的一部分
            'half': 横坐标, 纵坐标上个取一个像素加入目标

    Returns:
        Numpy n 维数组, 包含转换好的坐标和其它非坐标的输入值.
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1


def intersection_area(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    计算两组长方形边界框重叠的面积.

    假设 `boxes1` 和 `boxes2` 各包含 `m` 和 `n` 个边界框.

    如果 mode 为 'outer_product', 返回值为一个 `(m,n)` 的矩阵, 包含所有可能的`boxes1` 和 `boxes2`中边界框组合的边界框重叠的面积.

    如果 mode 为 'element-wise', `boxes1` and `boxes2` 的形状需要能够 broadcast. 

    Arguments:
        boxes1 (array): 要么形状为 `(4, )`, 包含安装 `coords` 指定的方式存储的一个边界框的坐标; 要么形状为 `(m, 4)`, 包含 `m` 个边界框.
            如果 `mode` 为 'element_wise', 形状必须能够和 `boxes2` broadcast.
        boxes2 (array): 包含安装 `coords` 指定的方式存储的一个边界框的坐标; 要么形状为 `(n, 4)`, 包含 `n` 个边界框.
            如果 `mode` 为 'element_wise', 形状必须能够和 `boxes1` broadcast.
        coords (str, optional): 输入数组的坐标格式:可以为 'centroids' `(cx, cy, w, h)`, 'minmax' `(xmin, xmax, ymin, ymax)`, 
            或者为 'corners' `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): 可以为 'outer_product' 或者 'element-wise'. 如果为 'outer_product' , 返回值为一个 `(m,n)` 的矩阵, 
            包含所有可能的`boxes1` 和 `boxes2`中边界框组合的边界框重叠的面积. 如果为 'element-wise' , 返回值为一个一维数组. 如果
            `boxes1` and `boxes2` 包含 `m` 个边界框, 返回值为一个一维数组, 其中第 i 个值为 `boxes1[i]` 与 `boxes2[i]`的边界框重叠面积.
        border_pixels (str, optional): 如何处理位于边界框上面的像素. 如何处理位于边界框上的像素. 取值可以为:
            'include': 边界上的像素属于边界框的一部分
            'exclude': 边界上的像素不属于边界框的一部分
            'half': 横坐标, 纵坐标上各取一个像素加入边界框

    Returns:
        一个一维或者二维 Numpy 数组, float 数据类型, 值为 `boxes1` 和 `boxes2` 中边界框的重叠面积.
    '''

    # 检测输入的参数的形状.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}: raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.",format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    m = boxes1.shape[0] # `boxes1` 中边界框的个数
    n = boxes2.shape[0] # `boxes2` 中边界框的个数

    # 根据设置的坐标格式, 设定对应坐标所处的位置.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # 边界上的像素计入边界框, 需要对 `xmax - xmin` 和 `ymax - ymin` 的值加一.
    elif border_pixels == 'exclude':
        d = -1 # 边界上的像素不计入边界框, 需要对 `xmax - xmin` 和 `ymax - ymin` 的值减一.

    # 计算重叠面积.

    if mode == 'outer_product':

        # 对于所有可能的组合, 获取较大的 xmin 和 ymin 的值.
        # min_xy 的形状为 (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))

        # 对于所有可能的组合, 获取较小的 xmax 和 ymax 的值.
        # max_xy 的形状为 (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))

        # 计算重叠区域的边长.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        # 计算重叠区域的边长.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]

def intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    '''
    与函数 'intersection_area()' 的功能相同, 为内部使用, 没有做输入参数的合法性检查
    '''

    m = boxes1.shape[0] 
    n = boxes2.shape[0] 

    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 
    elif border_pixels == 'exclude':
        d = -1 

    if mode == 'outer_product':

        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))

        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))

        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    计算两组矩形边界框的IoU.

    假设 `boxes1` and `boxes2` 分布包含 `m` 和 `n` 个边界框.

    如果 mode 为 'outer_product' , 返回一个形状为 `(m,n)` 的矩阵, 其中的值为 `boxes1` and `boxes2` 中所有可能的组合的IoU.

    如果 mode 为 'element-wise' , `m` 和 `n` 必须能够 broadcast.

    Arguments:
        boxes1 (array): 要么形状为 `(4, )`, 包含安装 `coords` 指定的方式存储的一个边界框的坐标; 要么形状为 `(m, 4)`, 包含 `m` 个边界框.
            如果 `mode` 为 'element_wise', 形状必须能够和 `boxes2` broadcast.
        boxes2 (array): 包含安装 `coords` 指定的方式存储的一个边界框的坐标; 要么形状为 `(n, 4)`, 包含 `n` 个边界框.
            如果 `mode` 为 'element_wise', 形状必须能够和 `boxes1` broadcast.
        coords (str, optional): 输入数组的坐标格式. 可以为 'centroids' `(cx, cy, w, h)`, 'minmax' `(xmin, xmax, ymin, ymax)`, 
            或者为 'corners' `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): 可以为 'outer_product' 或者 'element-wise'. 如果为 'outer_product' , 返回值为一个 `(m,n)` 的矩阵, 
            包含所有可能的`boxes1` 和 `boxes2`中边界框组合的IoU. 如果为 'element-wise' , 返回值为一个一维数组. 如果
            `boxes1` and `boxes2` 包含 `m` 个边界框, 返回值为一个一维数组, 其中第 i 个值为 `boxes1[i]` 与 `boxes2[i]`的IoU.
        border_pixels (str, optional): 如何处理位于边界框上面的像素. 如何处理位于边界框上的像素. 取值可以为:
            'include': 边界上的像素属于边界框的一部分
            'exclude': 边界上的像素不属于边界框的一部分
            'half': 横坐标, 纵坐标上各取一个像素加入边界框

    Returns:
        一个一维或者二维 Numpy 数组, float 数据类型, 值的范围在[0, 1]之间, 值为 `boxes1` 和 `boxes2` 中边界框的IoU. 
    '''

    # 确保输入的值的形状是正确的.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}: raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # 计算IoU.

    # 计算重叠面积.

    intersection_areas = intersection_area_(boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0] # `boxes1` 中边界框的个数
    n = boxes2.shape[0] # `boxes2` 中边界框的个数

    # 计算合并的面积.

    # 根据设置的坐标格式, 设定对应坐标所处的位置.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # 边界上的像素计入边界框, 需要对 `xmax - xmin` 和 `ymax - ymin` 的值加一.
    elif border_pixels == 'exclude':
        d = -1 # 边界上的像素不计入边界框, 需要对 `xmax - xmin` 和 `ymax - ymin` 的值减一.

    if mode == 'outer_product':

        boxes1_areas = np.tile(np.expand_dims((boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d), axis=1), reps=(1,n))
        boxes2_areas = np.tile(np.expand_dims((boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d), axis=0), reps=(m,1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d)
        boxes2_areas = (boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas
