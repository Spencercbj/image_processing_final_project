"""One-off: PIL resize 1440 -> Restormer -> ESRGAN+GFPGAN for photo 15."""
import os, sys, shutil, subprocess
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ESRGAN = os.path.join(BASE, 'Real-ESRGAN')
sys.path.insert(0, os.path.join(BASE, 'code'))

src = os.path.join(BASE, 'photos', '15_Photographer_Reflected_In_Night_Glass.jpg')
out_dir = os.path.join(BASE, 'results', 'original_15_face')
os.makedirs(out_dir, exist_ok=True)

# Step 1: PIL resize
img = Image.open(src)
w, h = img.size
sf = 1440 / max(w, h)
new_w, new_h = int(w * sf), int(h * sf)
img = img.resize((new_w, new_h), Image.LANCZOS)
resize_path = os.path.join(out_dir, '15_resize.png')
img.save(resize_path)
print(f'Resize: {w}x{h} -> {new_w}x{new_h}')

# Step 2: Restormer
from Restormer.inference import load_model, tile_inference
net = load_model(os.path.join(BASE, 'Restormer', 'Motion_Deblurring',
                              'pretrained_models', 'motion_deblurring.pth'), 'cuda')
img_cv = cv2.imread(resize_path)
rh, rw = img_cv.shape[:2]
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
pad_h = (8 - rh % 8) % 8
pad_w = (8 - rw % 8) % 8
if pad_h > 0 or pad_w > 0:
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
out = tile_inference(net, img_t, tile_size=896, overlap=128, device='cuda')
out = out[:, :, :rh, :rw].squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
out_bgr = (cv2.cvtColor(out, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
rest_path = os.path.join(out_dir, '15_restormer.png')
cv2.imwrite(rest_path, out_bgr)
print('Restormer done')
del net
torch.cuda.empty_cache()

# Step 3: ESRGAN + face enhance
tmp_in = os.path.join(out_dir, '_tmp_in2')
os.makedirs(tmp_in, exist_ok=True)
shutil.copy2(rest_path, os.path.join(tmp_in, '15.png'))
cmd = [
    sys.executable, os.path.join(ESRGAN, 'inference_realesrgan.py'),
    '-n', 'RealESRGAN_x4plus',
    '-i', tmp_in, '-o', out_dir,
    '-s', '4', '--tile', '256', '--suffix', 'resize_rest_face',
    '--face_enhance',
]
print('Running ESRGAN + face_enhance...')
subprocess.run(cmd, cwd=ESRGAN)
shutil.rmtree(tmp_in, ignore_errors=True)
print(f'Done: {os.path.join(out_dir, "15_resize_rest_face.png")}')
