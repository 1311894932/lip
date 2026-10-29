# Lip Sync

唇形同步 Gradio 演示：上传视频 → 自动检测人脸 → 选择人物 → 生成对口型视频。

## 环境

- Python 3.11，NVIDIA GPU（生成时显存需要 13G 以上）
- 系统里需要有 `ffmpeg` 和 `ffprobe`

```bash
pip install -r requirements.txt
```

## 权重

权重直接读取共享存储上的绝对路径，不需要下载。如果你的机器上共享存储挂载在别的位置，修改这三处：

| 权重 | 位置 |
|---|---|
| Dit模型、T5、VAE、whisper、wav2vec2 | `gradio_app.py` 的 `CHECKPOINTS_DIR` |
| ArcFace | `face_tracking.py` 的 `ARCFACE_CKPT_PATH` |
| DWPose ONNX | `third_party/annotator/dwpose/wholebody.py` 的 `ckpt_dir` |

## 运行

```bash
CUDA_VISIBLE_DEVICES=0 python gradio_app.py --port 7860
```

浏览器打开 `http://<服务器IP>:7860`：

1. 上传视频，自动检测第 0 帧的人脸（检测不到会显示 No Face Detected）
2. 用 Face Index 选择人物，可点 Preview Selected Face 预览
3. 默认使用视频自带的音轨；取消勾选 Use audio from reference video 则使用上传的音频
4. 点击 Generate Lip Sync，结果保存在 `./results/<视频名>_<时间戳>/`

其他参数：`--share` 生成公网链接；`--final_output_path <目录>` 把每次的结果额外复制一份到该目录。

如果启动时报 `startup-events failed (code 502)`，是终端设置了 HTTP 代理，先执行 `export NO_PROXY=127.0.0.1,localhost` 再启动。

第三方代码的来源和许可证见 `third_party/README.md`。
