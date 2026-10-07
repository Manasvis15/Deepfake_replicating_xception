"""
Self-contained re-implementation of DeepfakeBench's Xception deepfake detector.
No DeepfakeBench code needed: just PyTorch, OpenCV, NumPy (and scikit-learn for --eval).

It rebuilds the SAME network as training/networks/xception.py (mode='original'),
using the SAME layer names, so xception_best.pth loads with strict=True.

Commands
  python xception_standalone.py inspect --weights xception_best.pth
  python xception_standalone.py predict --weights xception_best.pth --input face.png
  python xception_standalone.py predict --weights xception_best.pth --input faces_folder/
  python xception_standalone.py eval    --weights xception_best.pth --real real_dir/ --fake fake_dir/

Inputs must be FACE CROPS (the model was trained on aligned faces, not full frames).
"""
import argparse, glob, os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ----------------------------------------------------------------------------------
# 1. ARCHITECTURE  (same as training/networks/xception.py, trimmed to mode='original')
# ----------------------------------------------------------------------------------
class SeparableConv2d(nn.Module):
    """Depthwise 3x3 conv (one filter per channel) followed by a 1x1 pointwise conv."""
    def __init__(self, cin, cout, k=1, stride=1, pad=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cin, k, stride, pad, dilation, groups=cin, bias=bias)
        self.pointwise = nn.Conv2d(cin, cout, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    """Residual unit: [ReLU -> SepConv -> BN] x reps (+ MaxPool if strided) + skip."""
    def __init__(self, cin, cout, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        if cout != cin or stride != 1:
            self.skip = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(cout)
        else:
            self.skip = None

        relu = nn.ReLU(inplace=True)
        rep, filters = [], cin
        if grow_first:
            rep += [relu, SeparableConv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout)]
            filters = cout
        for _ in range(reps - 1):
            rep += [relu, SeparableConv2d(filters, filters, 3, 1, 1), nn.BatchNorm2d(filters)]
        if not grow_first:
            rep += [relu, SeparableConv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout)]

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)   # must not overwrite the input used by the skip
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        # Index positions inside this Sequential must match the original, because they
        # become part of the weight names (e.g. "block1.rep.0.conv1.weight").
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        skip = self.skipbn(self.skip(inp)) if self.skip is not None else inp
        return x + skip


class Xception(nn.Module):
    def __init__(self, num_classes=2, inc=3):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2)
        self.block3 = Block(256, 728, 2, 2)
        # Middle flow: 8 identical blocks (block4 ... block11)
        for i in range(4, 12):
            setattr(self, f"block{i}", Block(728, 728, 3, 1))
        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = nn.Linear(2048, num_classes)
        # Unused in 'original' mode, but the original class always creates it, so its
        # weights are inside xception_best.pth. Keeping it lets strict loading succeed.
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=False))

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))           # [B,  32, 127, 127]
        x = self.relu(self.bn2(self.conv2(x)))           # [B,  64, 125, 125]
        for i in range(1, 13):                           # blocks 1..12
            x = getattr(self, f"block{i}")(x)            # -> [B, 1024, 8, 8]
        x = self.relu(self.bn3(self.conv3(x)))           # [B, 1536, 8, 8]
        x = self.bn4(self.conv4(x))                      # [B, 2048, 8, 8]
        return x

    def classifier(self, feat):
        x = F.adaptive_avg_pool2d(self.relu(feat), 1).flatten(1)   # [B, 2048]
        self.last_emb = x                                          # embedding for PCA/t-SNE
        return self.last_linear(x)                                 # [B, 2] logits

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat), feat


# ----------------------------------------------------------------------------------
# 2. LOADING THE .pth
# ----------------------------------------------------------------------------------
def load_state(weights):
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def build_model(weights, device):
    model = Xception()
    sd = load_state(weights)
    # xception_best.pth is the state_dict of XceptionDetector, whose network lives in
    # self.backbone, so every key starts with "backbone.". Multi-GPU (DDP) training adds
    # "module." in front. Strip both so the keys match this bare network.
    clean = {}
    for k, v in sd.items():
        for prefix in ("module.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    model.load_state_dict(clean, strict=True)   # errors loudly if anything doesn't match
    print(f"[ok] loaded {len(clean)} tensors from {weights}")
    return model.to(device).eval()               # eval(): BatchNorm uses stored statistics


# ----------------------------------------------------------------------------------
# 3. PREPROCESSING  (from config/detector/xception.yaml and dataset code)
# ----------------------------------------------------------------------------------
def preprocess(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0   # [3,256,256] in [0,1]
    return (x - 0.5) / 0.5                                         # mean=std=0.5 -> [-1,1]


@torch.no_grad()
def predict(model, paths, device, bs=32):
    probs = []
    for i in range(0, len(paths), bs):
        x = torch.stack([preprocess(p) for p in paths[i:i + bs]]).to(device)
        logits, _ = model(x)
        probs.append(F.softmax(logits, dim=1)[:, 1].cpu())        # class 1 = fake
    return torch.cat(probs).numpy()


def list_images(p):
    if os.path.isdir(p):
        return sorted(f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True)
                      if f.lower().endswith(IMG_EXT))
    return [p]


# ----------------------------------------------------------------------------------
# 4. COMMANDS
# ----------------------------------------------------------------------------------
def cmd_inspect(a):
    sd = load_state(a.weights)
    print(f"{len(sd)} tensors. First 15 keys:")
    for k in list(sd)[:15]:
        print(f"  {k:55s} {tuple(sd[k].shape)}")
    print("  ...")
    for k in list(sd)[-4:]:
        print(f"  {k:55s} {tuple(sd[k].shape)}")


def cmd_predict(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(a.weights, device)
    paths = list_images(a.input)
    for p, s in zip(paths, predict(model, paths, device)):
        print(f"{p}: P(fake)={s:.4f} -> {'FAKE' if s >= a.threshold else 'REAL'}")


def cmd_eval(a):
    from sklearn.metrics import roc_auc_score, accuracy_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(a.weights, device)
    real, fake = list_images(a.real), list_images(a.fake)
    scores = np.concatenate([predict(model, real, device), predict(model, fake, device)])
    labels = np.array([0] * len(real) + [1] * len(fake))
    print(f"real={len(real)} fake={len(fake)}")
    print(f"frame-level AUC = {roc_auc_score(labels, scores):.4f}")
    print(f"accuracy@{a.threshold} = {accuracy_score(labels, scores >= a.threshold):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "predict", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--weights", required=True)
        s.add_argument("--threshold", type=float, default=0.5)
        if name == "predict":
            s.add_argument("--input", required=True)
        if name == "eval":
            s.add_argument("--real", required=True)
            s.add_argument("--fake", required=True)
    a = ap.parse_args()
    {"inspect": cmd_inspect, "predict": cmd_predict, "eval": cmd_eval}[a.cmd](a)