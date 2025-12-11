# Basketball High Light Detection Project Report LaTex & PPT

推荐使用 [YanMing-lxb/PyTeXMK](https://github.com/YanMing-lxb/PyTeXMK) 进行编译

## TODO

### 2025-12-18

- [] 继续优化连续性
- [] 统一多视角
- [] 利用多视角提升球追踪效果
- [] 利用多视角、根据人的位置连续性、不突变特性优化提升人物追踪，人脸检索效果
- [] 二维重建

### 2025-12-11

- [x] 下好数据集，处理好，训练 yolo12x
- [x] 完善 PPT 内容，补充更多训练细节（12.7 Added）
- [x] 对比解决畸变前后的效果，对比训练前后的效果，给出数据统计图对比
- [x] 尝试使用上次说的连续性，或者统一两个视角解决人脸有的视角下不出现的问题，给出一个帧级 acc
- []（可选）研究根据两个视角的二维重建
