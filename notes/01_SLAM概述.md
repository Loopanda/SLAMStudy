## 01_SLAM概述

#### 1. 传感器分类

举个例子：我们可以在房间地板上铺设导引线，在墙壁上贴识别二维码，在桌子上放置无线电定位设备。如果在室外，还可以在机器人上安装 GPS定位设备。

<img src="C:\Users\86150\AppData\Roaming\Typora\typora-user-images\image-20221126163910579.png" alt="image-20221126163910579" style="zoom:50%;" />

- 携带于机器人本体上：例如机器人的轮式编码器、相机、激光等等
- 安装于环境中：导轨、二维码标志等等

#### 2. 相机分类

单目(Monocular)、双目(Stereo)和深度相机(RGB-D)三个大类

<img src="C:\Users\86150\AppData\Roaming\Typora\typora-user-images\image-20221126164049952.png" alt="image-20221126164049952" style="zoom:70%;" />

##### 2.1 单目相机

优势：传感器结构特别的简单、成本特别的低

形式：相机的成像平面上留下的一个投影，它以二维的形式反映了三维的世界

缺陷：丢掉了场景的一个维度：也就是所谓的深度(或距离)，具有尺度不确定性

识别深度的方式：恢复三维结构，必须移动相机的视角。在单目中也是同样的原理。我们必须移动相机之后，才能估计它的运动

尺度：单目估计的轨迹和地图，将与真实的轨迹、地图，相差一个因子，也就是所谓的尺度(Scale)

##### 2.2 双目相机

组成：双目相机由两个单目相机组成，但这两个相机之间的距离(基线)是已知的。

计算方式：基线距离越大，能够测量到的距离就越远

优势：是比较左右眼的图像获得的，并不依赖其他传感设备，所以它既可以应用在室内，亦可应用于室外

缺陷：其深度量程和精度受双目的基线与分辨率限制，而且视差的计算非常消耗计算资源

##### 2.3 深度相机

计算方式：通过主动向物体发射光并接收返回的光，测出物体离相机的距离

缺陷：测量范围窄、噪声大、视野小、易受日光干扰、无法测量透射材质等

应用场景主要在室内室外较难运用

#### 3. SLAM整体框架

<img src="C:\Users\86150\AppData\Roaming\Typora\typora-user-images\image-20221126164946049.png" alt="image-20221126164946049" style="zoom:80%;" />

- 传感器信息读取。在视觉SLAM 中主要为相机图像信息的读取和预处理。如果在机器人中，还可能有码盘、惯性传感器等信息的读取和同步。
- 视觉里程计(Visual Odometry, VO)。视觉里程计任务是估算相邻图像间相机的运动，以及局部地图的样子。VO 又称为前端（Front End）。
- 后端优化（Optimization）。后端接受不同时刻视觉里程计测量的相机位姿，以及回环检测的信息，对它们进行优化，得到全局一致的轨迹和地图。由于接在VO 之后，又称为后端（Back End）。
- 回环检测（Loop Closing）。回环检测判断机器人是否曾经到达过先前的位置。如果检测到回环，它会把信息提供给后端进行处理。
- 建图（Mapping）。它根据估计的轨迹，建立与任务要求对应的地图。

##### 3.1 视觉里程计

视觉里程计关心相邻图像之间的相机运动，VO能够通过相邻帧间的图像估计相机运动，并恢复场景的空间结构

里程计没解决的问题：仅通过视觉里程计来估计轨迹，每次估计都带有一定的误差，将不可避免地出现**累计漂移**。先前时刻的误差将会传递到下一时刻，导致经过一段时间之后，估计的轨迹将不再准确

##### 3.2 后端优化

主要指处理过程中噪声的问题，前端和计算机视觉研究领域更为相关，比如图像的特征提取与匹配等，后端则主要是滤波与非线性优化算法

##### 3.3 回环检测

主要解决位置估计随时间漂移的问题，如果有某种手段，让机器人知道“回到了原点”这件事，或者把“原点”识别出来，我们再把位置估计值“拉”过去，就可以消除漂移了，这就是所谓的回环检测

为了实现回环检测，我们需要让机器人具有识别曾到达过的场景的能力

检测方法：可以判断图像间的相似性，来完成回环检测

##### 3.4 建图

地图是对环境的描述，但这个述并不是固定的，需要视的SLAM应用而定。

<img src="C:\Users\86150\AppData\Roaming\Typora\typora-user-images\image-20221126165916206.png" alt="image-20221126165916206" style="zoom:67%;" />

###### 3.4.1 度量地图

精确地表示地图中物体的位置关系，通常我们用稀疏与稠密对它们进行分类

缺陷：这种地图需要存储每一个格点的状态，耗费大量的存储空间，而且多数情况下地图的许多细节部分是无用

的

###### 3.4.2 拓扑地图

拓扑地图则更强调地图元素之间的关系

它放松了地图对精确位置的需要，去掉地图的细节问题，是一种更为紧凑的表达方式。然而，拓扑地图不擅长表达具有复杂结构的地图。

#### 4. SLAM数学建模

见P24