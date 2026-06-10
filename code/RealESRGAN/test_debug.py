import os, sys, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Real-ESRGAN'))
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

base = os.path.join(os.path.dirname(__file__), '..', '..')
model_path = os.path.join(base, 'Real-ESRGAN', 'weights', 'RealESRGAN_x4plus.pth')
inp = os.path.join(base, 'results', '03_Restormer', '01.png')
out = os.path.join(base, 'results', '05_RealESRGAN', '01.png')

os.makedirs(os.path.dirname(out), exist_ok=True)

print(f'Model: {model_path} exists={os.path.exists(model_path)}')
print(f'Input: {inp} exists={os.path.exists(inp)}')

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=256, tile_pad=10, pre_pad=0, half=True)

img = cv2.imread(inp, cv2.IMREAD_UNCHANGED)
print(f'Image shape: {img.shape}')

output, _ = upsampler.enhance(img, outscale=1)
print(f'Output shape: {output.shape}')
cv2.imwrite(out, output)
print(f'Saved to {out}')
