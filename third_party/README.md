# Third-party code

Vendored so that `gradio_app.py` runs without sibling checkouts. `face_tracking.py`
puts these directories on `sys.path`.

| Directory | Origin | License | Changes made here |
|---|---|---|---|
| `annotator/dwpose/` | [DWPose](https://github.com/IDEA-Research/DWPose), `ControlNet-v1-1-nightly/annotator/dwpose` (ONNX inference), copied from the local checkout including its `DWposeDetector.get_lmks()` helper | Apache-2.0, see `annotator/LICENSE` | `wholebody.py`: ONNX weights are read from an absolute path on the shared storage |
| `arcface/backbones/` | [insightface](https://github.com/deepinsight/insightface), `recognition/arcface_torch/backbones` | MIT (code) | none |
| `arcface/id_loss.py` | ArcFace identity-embedding wrapper | — | trimmed to `ArcFaceLoss`; imports `backbones/` from its own directory |

## Weights

All weights are read from absolute paths on the shared storage (`/afs-mt/shanhai-ai/...`);
if it is mounted elsewhere, edit these three places:

| Weights | Set in |
|---|---|
| X-Dub DiT, umt5-xxl text encoder + tokenizer, Wan2.2 VAE, whisper large-v2, wav2vec2-base-960h | `CHECKPOINTS_DIR` in `gradio_app.py` |
| ArcFace r100 recognition backbone (`backbone.pth`) | `ARCFACE_CKPT_PATH` in `face_tracking.py` |
| DWPose ONNX (`yolox_l.onnx`, `dw-ll_ucoco_384.onnx`) | `ckpt_dir` in `third_party/annotator/dwpose/wholebody.py` |

The insightface pretrained recognition models are released for non-commercial research
purposes only (see the insightface README).
