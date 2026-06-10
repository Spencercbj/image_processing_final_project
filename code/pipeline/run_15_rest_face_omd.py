"""One-off: take 15_resize_rest_face.png -> PIL resize 1440 -> OMDNet -> ESRGAN+GFPGAN."""
import os, sys, shutil, subprocess
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as tf

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ESRGAN = os.path.join(BASE, 'Real-ESRGAN')
sys.path.insert(0, os.path.join(BASE, 'OMDNet'))
sys.path.insert(0, os.path.join(BASE, 'code'))

from models.deblur_model import OMDNet
from utils.general import move_to_cuda
from utils.record import load_checkpoint, plot_img

src = os.path.join(BASE, 'results', 'original_15_face', '15_resize_rest_face.png')
out_dir = os.path.join(BASE, 'results', 'original_15_face')

# Step 1: PIL resize to 1440
img = Image.open(src)
w, h = img.size
sf = 1440 / max(w, h)
new_w, new_h = int(w * sf), int(h * sf)
img = img.resize((new_w, new_h), Image.LANCZOS)
print(f'Resize: {w}x{h} -> {new_w}x{new_h}')

# Step 2: OMDNet
model = OMDNet(num_res=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
load_checkpoint(model, optimizer, None, os.path.join(BASE, 'OMDNet', 'checkpoints', 'test1'))
torch.set_grad_enabled(False)
model.eval()

img_np = np.uint8(np.array(img))
img_t = tf.to_tensor(img_np).unsqueeze_(0)
mw = 8
if img_t.shape[2] % mw != 0:
    img_t = F.pad(img_t, (0, 0, 0, mw - img_t.shape[2] % mw), mode='reflect')
if img_t.shape[3] % mw != 0:
    img_t = F.pad(img_t, (0, mw - img_t.shape[3] % mw, 0, 0), mode='reflect')

img_t = move_to_cuda(img_t)
_, deblured, _, _, _, _, _ = model(img_t, img_t, img_t, train=False)
omd_path = os.path.join(out_dir, '15_rest_face_omd.png')
plot_img(deblured[0][0]).save(omd_path)
print(f'OMDNet done: {omd_path}')
del model
torch.cuda.empty_cache()
torch.set_grad_enabled(True)

# Step 3: ESRGAN + face enhance
tmp_in = os.path.join(out_dir, '_tmp_in3')
os.makedirs(tmp_in, exist_ok=True)
shutil.copy2(omd_path, os.path.join(tmp_in, '15.png'))
cmd = [
    sys.executable, os.path.join(ESRGAN, 'inference_realesrgan.py'),
    '-n', 'RealESRGAN_x4plus',
    '-i', tmp_in, '-o', out_dir,
    '-s', '4', '--tile', '256', '--suffix', 'rest_face_omd_esrgan',
    '--face_enhance',
]
print('Running ESRGAN + face_enhance...')
subprocess.run(cmd, cwd=ESRGAN)
shutil.rmtree(tmp_in, ignore_errors=True)
print(f'Done: {os.path.join(out_dir, "15_rest_face_omd_esrgan.png")}')
