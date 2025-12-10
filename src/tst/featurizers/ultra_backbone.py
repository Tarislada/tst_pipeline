
from __future__ import annotations
import torch, torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from torchvision.ops import roi_align

try:
    import cv2
except Exception:
    cv2 = None

def resize_pad_to_square(img: np.ndarray, size: int = 640) -> Tuple[torch.Tensor, Tuple[float,float,int,int]]:
    h, w = img.shape[:2]
    if cv2 is None:
        raise ImportError("cv2 is required for resize_pad_to_square")
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = np.zeros((size, size, 3), dtype=np.uint8)
    img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    resized[pad_y:pad_y+nh, pad_x:pad_x+nw] = img_rs
    x = torch.from_numpy(resized).permute(2,0,1).float() / 255.0
    return x, (scale, scale, pad_x, pad_y)

class HookBank:
    def __init__(self, model: nn.Module, layer_paths: List[str]):
        self.model = model
        self.layer_paths = layer_paths
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: Dict[str, torch.Tensor] = {}
        self._register()

    def _resolve(self, path: str) -> nn.Module:
        m = self.model
        if path in ("", "<root>"):
            return m
        for key in path.split("."):
            if key.isdigit():
                m = m[int(key)]
            else:
                m = getattr(m, key)
        return m

    def _hook(self, name: str):
        def fn(_mod, _in, out):
            self.cache[name] = out.detach()
        return fn

    def _register(self):
        for p in self.layer_paths:
            mod = self._resolve(p)
            self.handles.append(mod.register_forward_hook(self._hook(p)))

    def clear(self):
        self.cache.clear()

    def close(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles = []

class UltralyticsSegFeaturizer:
    def __init__(self, yolo_model, layer_paths: List[str], imgsz: int = 640, device: str = "cuda"):
        self.model = yolo_model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.core = getattr(self.model, "model", self.model)
        self.hooks = HookBank(self.core, layer_paths)
        self.imgsz = imgsz

    @torch.inference_mode()
    def extract_batch(self, frames_bgr: List[np.ndarray], instance_boxes_xyxy: List[np.ndarray], instance_masks: List[np.ndarray]) -> np.ndarray:
        assert len(frames_bgr) == len(instance_boxes_xyxy) == len(instance_masks)
        B = len(frames_bgr)

        # Preprocess
        x_list, meta = [], []
        for img in frames_bgr:
            x, (sx, sy, px, py) = resize_pad_to_square(img, self.imgsz)
            x_list.append(x)
            meta.append((sx, sy, px, py))
        batch = torch.stack(x_list, 0).to(self.device)

        _ = self.model(batch, verbose=False)

        fmap_dict = {k: v for k, v in self.hooks.cache.items()}
        names = list(fmap_dict.keys())
        maps = [fmap_dict[n] for n in names]

        pooled_per_frame: List[np.ndarray] = []
        for i in range(B):
            boxes = instance_boxes_xyxy[i]
            masks = instance_masks[i]
            if boxes is None or len(boxes) == 0:
                continue

            layer_vecs = []
            for fmap in maps:
                fi = fmap[i:i+1]
                _, C, h, w = fi.shape
                sx, sy, px, py = meta[i]

                rois = []
                for (x1, y1, x2, y2) in boxes:
                    lx1 = x1 * sx + px
                    ly1 = y1 * sy + py
                    lx2 = x2 * sx + px
                    ly2 = y2 * sy + py
                    fx = w / float(self.imgsz)
                    fy = h / float(self.imgsz)
                    rx1, ry1, rx2, ry2 = lx1 * fx, ly1 * fy, lx2 * fx, ly2 * fy
                    rois.append([0.0, rx1, ry1, rx2, ry2])
                rois = torch.tensor(rois, dtype=torch.float32, device=fi.device)

                aligned = roi_align(fi, rois, output_size=(7,7), spatial_scale=1.0, aligned=True)
                vec = aligned.flatten(2).mean(-1)
                layer_vecs.append(vec)

            feat = torch.cat(layer_vecs, dim=1)
            pooled_per_frame.append(feat.detach().cpu().numpy())

        if not pooled_per_frame:
            total_c = sum(m.shape[1] for m in maps) if maps else 0
            return np.zeros((0, total_c), dtype=np.float32)
        return np.concatenate(pooled_per_frame, axis=0)

    def close(self):
        self.hooks.close()
