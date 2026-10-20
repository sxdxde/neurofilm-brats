"""FiLM-conditioned U-Net definition

Extracted verbatim from the notebook. Shares variables with earlier
stages, so run via run_all.py (shared namespace) rather than standalone.

## Cell 12 — FiLM-Conditioned U-Net Architecture Definition

Defines all model classes used throughout the notebook. **No external segmentation library is needed.**

### Architecture: FiLM U-Net

A standard U-Net encoder–decoder with **Feature-wise Linear Modulation (FiLM)** injected at the bottleneck and every decoder stage.

**How FiLM works:**
Given a 512-D text embedding `e`, FiLM learns per-channel scale `γ` and bias `β`:
```
output = (1 + γ(e)) · feature_map + β(e)
```
This lets the radiology text dynamically amplify or suppress specific feature channels in the decoder — for example, boosting channels that respond to the tumour morphology described in the report.

**Image-only baseline:** pass `emb=None` (or zero embeddings). All FiLM layers become identity functions (γ initialised to 0), giving an architecturally identical baseline with the same parameter count.

### Channel progression
```
Encoder:  1 → 32 → 64 → 128 → 256
Bottleneck:        256 → 512
Decoder:  512 → 256 → 128 → 64 → 32 → 1 (sigmoid)
```
"""

# ── Step C: FiLM-conditioned U-Net (fully inline, no external imports) ────────
import torch, torch.nn as nn, torch.nn.functional as F

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: output = (1+gamma) * x + beta"""
    def __init__(self, text_dim, channels):
        super().__init__()
        self.fc = nn.Linear(text_dim, 2*channels)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self, x, emb):
        gb = self.fc(emb)
        g, b = gb.chunk(2, dim=1)
        return (1 + g.view(-1,x.shape[1],1,1)) * x + b.view(-1,x.shape[1],1,1)

class ConvBnRelu(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class FiLMUNet(nn.Module):
    """
    U-Net with FiLM text conditioning at bottleneck + all decoder stages.
    text_emb=None or zeros → image-only baseline (FiLM layers are identity when zeroed).
    """
    def __init__(self, text_dim=512):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBnRelu(1,   32)
        self.e2 = ConvBnRelu(32,  64)
        self.e3 = ConvBnRelu(64,  128)
        self.e4 = ConvBnRelu(128, 256)
        self.bn = ConvBnRelu(256, 512)

        self.film_bn = FiLM(text_dim, 512)
        self.film_d4 = FiLM(text_dim, 256)
        self.film_d3 = FiLM(text_dim, 128)
        self.film_d2 = FiLM(text_dim,  64)
        self.film_d1 = FiLM(text_dim,  32)

        self.u4 = nn.ConvTranspose2d(512,256,2,stride=2); self.d4 = ConvBnRelu(512,256)
        self.u3 = nn.ConvTranspose2d(256,128,2,stride=2); self.d3 = ConvBnRelu(256,128)
        self.u2 = nn.ConvTranspose2d(128, 64,2,stride=2); self.d2 = ConvBnRelu(128, 64)
        self.u1 = nn.ConvTranspose2d( 64, 32,2,stride=2); self.d1 = ConvBnRelu( 64, 32)
        self.head = nn.Conv2d(32,1,1)

    def forward(self, x, emb=None):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b  = self.bn(self.pool(s4))

        if emb is not None:
            b  = self.film_bn(b, emb)

        d4 = self.d4(torch.cat([self.u4(b),  s4], 1))
        if emb is not None: d4 = self.film_d4(d4, emb)
        d3 = self.d3(torch.cat([self.u3(d4), s3], 1))
        if emb is not None: d3 = self.film_d3(d3, emb)
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        if emb is not None: d2 = self.film_d2(d2, emb)
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        if emb is not None: d1 = self.film_d1(d1, emb)
        return self.head(d1)

# Sanity check
device = get_device()   # MPS / CUDA / CPU — defined in Cell 1
_m = FiLMUNet().to(device)
_x = torch.randn(2,1,128,128,device=device)
_e = torch.randn(2,512,device=device)
_o = _m(_x, _e)
print(f"FiLMUNet output: {_o.shape}")
print(f"Parameters: {sum(p.numel() for p in _m.parameters()):,}")
del _m, _x, _e, _o; empty_cache(device)

