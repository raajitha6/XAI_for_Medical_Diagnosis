import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CT_PATH = "volume-28.nii"
MASK_PATH = "segmentation-28.nii"  # only used for patch selection


# ===== MODEL =====
def conv_block(in_c,out_c):
    return nn.Sequential(
        nn.Conv3d(in_c,out_c,3,padding=1),
        nn.BatchNorm3d(out_c),
        nn.ReLU(),
        nn.Conv3d(out_c,out_c,3,padding=1),
        nn.BatchNorm3d(out_c),
        nn.ReLU()
    )

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = conv_block(1,32)
        self.e2 = conv_block(32,64)
        self.e3 = conv_block(64,128)
        self.pool = nn.MaxPool3d(2)
        self.b = conv_block(128,256)
        self.u3 = nn.ConvTranspose3d(256,128,2,2)
        self.d3 = conv_block(256,128)
        self.u2 = nn.ConvTranspose3d(128,64,2,2)
        self.d2 = conv_block(128,64)
        self.u1 = nn.ConvTranspose3d(64,32,2,2)
        self.d1 = conv_block(64,32)
        self.out = nn.Conv3d(32,1,1)

    def forward(self,x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b),e3],1))
        d2 = self.d2(torch.cat([self.u2(d3),e2],1))
        d1 = self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1)


# ===== LOAD MODEL =====
model = UNet3D().to(DEVICE)
checkpoint = torch.load("best_model_final.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ===== UTILS =====
def load_nii(path):
    return nib.load(path).get_fdata()

def normalize(ct):
    ct = np.clip(ct, -200, 300)
    ct = (ct + 200) / 500
    return ct.astype(np.float32)

def pad_to_size(img, target_shape):
    pad_width = []
    for i in range(3):
        diff = target_shape[i] - img.shape[i]
        pad_width.append((0, max(diff,0)))
    return np.pad(img, pad_width, mode='constant')

def extract_patch(ct, mask=None, size=(96,96,48)):
    x,y,z = ct.shape
    px,py,pz = size

    if mask is not None:
        coords = np.where(mask==2)
        idx = len(coords[0])//2
        cx,cy,cz = coords[0][idx], coords[1][idx], coords[2][idx]
    else:
        cx,cy,cz = x//2, y//2, z//2

    x1 = max(cx - px//2, 0)
    y1 = max(cy - py//2, 0)
    z1 = max(cz - pz//2, 0)

    patch = ct[x1:x1+px, y1:y1+py, z1:z1+pz]
    return pad_to_size(patch, size)


# ===== MAIN FUNCTION =====
def run():

    ct = normalize(load_nii(CT_PATH))

    try:
        mask = load_nii(MASK_PATH)
        ct_patch = extract_patch(ct, mask)
    except:
        ct_patch = extract_patch(ct)

    x = torch.tensor(ct_patch).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0,0].cpu().numpy()

    threshold = 0.5
    pred_bin = (pred > threshold).astype(float)

    slice_idx = ct_patch.shape[-1] // 2

    ct_slice = ct_patch[:,:,slice_idx]
    pred_slice = pred_bin[:,:,slice_idx]

    # ===== GRAD-CAM =====
    activations = None
    gradients = None

    def f_hook(m,i,o):
        nonlocal activations
        activations = o

    def b_hook(m,gi,go):
        nonlocal gradients
        gradients = go[0]

    h1 = model.e3.register_forward_hook(f_hook)
    h2 = model.e3.register_backward_hook(b_hook)

    model.zero_grad()
    out = model(x)
    torch.sigmoid(out).mean().backward()

    acts = activations.detach().cpu().numpy()[0]
    grads = gradients.detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1,2,3))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i,w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam,0)
    cam = (cam - cam.min())/(cam.max()+1e-8)

    scale = (
        pred.shape[0]/cam.shape[0],
        pred.shape[1]/cam.shape[1],
        pred.shape[2]/cam.shape[2]
    )

    cam = zoom(cam, scale, order=1)

    h1.remove()
    h2.remove()

    cam_slice = cam[:,:,slice_idx]

    # ===== MATPLOTLIB FIGURE =====
    fig, axes = plt.subplots(1,3, figsize=(12,4))

    # CT
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("CT")
    axes[0].axis("off")

    # Prediction
    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].imshow(pred_slice, alpha=0.4, cmap="Blues")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Grad-CAM
    axes[2].imshow(ct_slice, cmap="gray")
    axes[2].imshow(cam_slice, alpha=0.5, cmap="jet")

    # contour like your notebook
    axes[2].contour(pred_slice, colors='cyan', linewidths=1)

    axes[2].set_title("Grad-CAM")
    axes[2].axis("off")

    plt.tight_layout()

    return fig


# ===== GRADIO UI =====
app = gr.Interface(
    fn=run,
    inputs=[],
    outputs=gr.Plot(),
    title="Explainable 3D Liver Tumor Segmentation"
)

app.launch()