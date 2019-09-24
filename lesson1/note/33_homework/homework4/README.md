# Project 3：自动驾驶

1. task1 完成如下函数：

   * horizontal_flip(img, degree)  

     * 按照50%的概率水平翻转图像
     * img: 输入图像
     * degree: 输入图像对于的转动角度

   *  random_brightness(img, degree)

     * 随机调整输入图像的亮度， 调整强度于 0.1(变黑)和1(无变化)之间
     * img: 输入图像
     * degree: 输入图像对于的转动角度

   *  left_right_random_swap(img_address, degree, degree_corr=1.0 / 4)

     * 随机从左， 中， 右图像中选择一张图像， 并相应调整转动的角度
     * img_address: 中间图像的的文件路径
     * degree: 中间图像对于的方向盘转动角度
     * degree_corr: 方向盘转动角度调整的值

   * discard_zero_steering(degrees, rate)

     * 从角度为零的index中随机选择部分index返回
     * degrees: 输入的角度值
     * rate: 丢弃率， 如果rate=0.8， 意味着80%的index会被返回， 用于丢弃

   * image_transformation(img_address, degree, data_dir)

     * 图像整体预处理（待完善）

     * img_address: 中间图像的的文件路径

     * degree：中间图像对于的方向盘转动角度

     * data_dir：图像数据路径

       

2. task2 构造自己的网络结构
   * get_model(shape)
     * 预测方向盘角度: 以图像为输入, 预测方向盘的转动角度（待完善）
     * shape: 输入图像的尺寸, 例如(128, 128, 3)

3. task3 自己调试自动的网络, 录制一个视频, 显示自己训练的小车能够自动驾驶整个模拟器的赛道