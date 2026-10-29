import torch
import torch.nn.functional as F
import os

# insightface arcface_torch backbones/ is vendored next to this file; face_tracking.py puts
# this directory on sys.path, so it imports as a top-level package.
try:
    from backbones import get_model
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False
    print("Warning: arcface_torch backbones not found next to id_loss.py.")

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ArcFaceLoss(torch.nn.Module):
    """
    ArcFace 身份损失 (可导 pytorch backbone)
    权重: glint360k_cosface_r100 的 backbone.pth, key 完全匹配 get_model('r100')
    """
    def __init__(self, ckpt_path=None, device='cuda', backbone='r100',
                 input_is_bgr=False):
        super().__init__()
        self.device = device
        self.backbone = backbone
        self.input_is_bgr = input_is_bgr  # 见 preprocess 说明

        if not ARCFACE_AVAILABLE:
            raise ImportError("arcface_torch backbones not available.")

        self.net = get_model(backbone, fp16=False)

        if ckpt_path is None:
            ckpt_path = self._download_arcface_weights(backbone)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ArcFace checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location='cpu')
        # 有些权重包成 {'state_dict': ...} 或带 module. 前缀，做一下兼容
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        # 加载后必须检查：missing / unexpected 应该基本为空
        print(f"[ArcFace] load r100: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 5 or len(unexpected) > 5:
            print("[ArcFace][WARN] 权重 key 大量不匹配，可能用错了权重(非识别 backbone)!")

        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _download_arcface_weights(backbone='r100'):
        """从 HuggingFace 下载可导的 arcface r100 识别 backbone"""
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub not installed.")
        if backbone != 'r100':
            raise ValueError(f"Unsupported backbone: {backbone}")
        # glint360k_cosface_r100 backbone.pth，社区镜像之一
        # 若该 repo 失效，换成你能访问的同类 repo 即可，要点是 r100 识别 backbone
        local_path = hf_hub_download(
            repo_id="gaunernst/arcface-r100-glint360k",  # 见下方说明，可替换
            filename="backbone.pth",
        )
        return local_path

    def preprocess(self, x):
        """
        输入: b c h w, [-1,1], RGB
        ArcFace 训练管线: BGR, (x-127.5)/127.5 即 [-1,1]
        - 你的输入已是 [-1,1]，无需再做 mean/std
        - 通道: insightface 习惯 BGR；input_is_bgr=False 时这里翻转一下
        """
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        if not self.input_is_bgr:
            x = x.flip(1)  # RGB -> BGR
        return x  # 仍在 [-1,1]

    def forward(self, pred, ref, return_similarity=False):
        # 确保输入数据类型与网络权重一致 (float32)
        pred = pred.float()
        ref = ref.float()
        fp = self.net(self.preprocess(pred))
        fr = self.net(self.preprocess(ref))
        cos = F.cosine_similarity(fp, fr, dim=-1)
        if return_similarity:
            return cos.mean()
        return (1.0 - cos).mean()
