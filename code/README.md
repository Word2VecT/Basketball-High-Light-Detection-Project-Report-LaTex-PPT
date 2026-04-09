# Basketball Highlight Detection

此项目包含篮球轨迹追踪的流水线代码。

## 项目结构

- `src/track/`: 源代码目录
  - `pipeline.py`: 主处理流水线
  - `traj_*.py`: 轨迹生成、平滑、匹配、ReID等模块
  - `siglip.py`: SigLIP 模型封装
- `assets/`: 资源文件
  - `court__bg.png`: 球场背景图
  - `homo/`: 单应性矩阵文件
  - `ref/`: 参考图片

## 使用方法

此项目使用 `uv` 进行依赖管理。

1. 安装依赖：
   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. 运行流水线：
   在项目根目录下运行：
   ```bash
   HF_ENDPOINT="https://hf-mirror.com" python -m src.track.pipeline
   ```

<br />

## TODO

- [ ] 精准测算一个流程的时间 现在实验室的服务器比较卡 测出的时间有问题（现在300f*3为6min半 我怀疑异步本身没有合适地工作） （理应200f的轨迹生成是20多秒，绝对超过了其他的环节时间，所以处理一段 异步总共20多秒）可以参考demo 文件夹下的face_recognition_demo.py 里面正常时 200帧是17-23s不等，现在服务器跑出来都1min了
- [ ] 不依靠YOLO的追踪 只用检测+reid （检测到某处有某个id，就把这个点画上去，最后把某个id的轨迹全部整理）
- [ ] 参数是否需要调优 轨迹融合 片段融合还是存在缺陷 有时候会少几个轨迹
- [ ] 3维重建（optional）
- [ ] 能否减小person roi的大小 把上半身（或更小）从检测框中裁出来，交给face_analyser 是否可以提高速度

<br />

## 注意事项

- 代码中的文件路径已更新为相对于项目根目录的 `assets/` 路径。请确保在项目根目录下运行脚本。
- 如需修改配置，请参考 `pipeline.py` 中的配置部分。
- 参数的配置说明位于

