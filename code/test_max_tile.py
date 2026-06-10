"""Find max tile size for each model on this GPU."""
import sys, os, torch, gc

BASE = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(BASE, 'DarkIR'))
sys.path.insert(0, os.path.join(BASE, 'Restormer'))
sys.path.insert(0, os.path.join(BASE, 'NAFNet'))

device = 'cuda'
sizes = [384, 512, 640, 768, 896, 1024, 1152, 1280]

def test_model(name, build_fn, forward_fn):
    print(f'\n=== {name} ===')
    net = build_fn().eval().to(device)
    best = 0
    for s in sizes:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            x = torch.randn(1, 3, s, s, device=device)
            with torch.no_grad():
                _ = forward_fn(net, x)
            vram = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
            print(f'  {s}x{s}: OK  (peak {vram:.2f} GB)')
            best = s
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'  {s}x{s}: OOM')
                torch.cuda.empty_cache()
                break
            else:
                raise
    print(f'  -> Max tile: {best}x{best}')
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return best


# NAFNet
from basicsr.models.archs.NAFNet_arch import NAFNet
def build_nafnet():
    net = NAFNet(img_channel=3, width=64, middle_blk_num=1,
                 enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])
    s = torch.load(os.path.join(BASE, 'NAFNet/experiments/pretrained_models/NAFNet-REDS-width64.pth'),
                   map_location='cpu')
    net.load_state_dict(s['params'], strict=True)
    return net

test_model('NAFNet (REDS w64)', build_nafnet, lambda net, x: net(x))


# DarkIR
from archs.DarkIR import DarkIR
def build_darkir():
    net = DarkIR(img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2,
                 enc_blk_nums=[1,2,3], dec_blk_nums=[3,1,1],
                 dilations=[1,4,9], extra_depth_wise=True)
    ckpt = torch.load(os.path.join(BASE, 'DarkIR/models/DarkIR_384.pt'),
                      map_location='cpu', weights_only=False)
    state = ckpt.get('params', ckpt.get('model_state_dict', ckpt))
    cleaned = {k.replace('module.',''): v for k, v in state.items()}
    net.load_state_dict(cleaned, strict=True)
    return net

test_model('DarkIR (w32)', build_darkir, lambda net, x: net(x, side_loss=False))


# Restormer — need to bypass NAFNet's installed basicsr
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "restormer_arch",
    os.path.join(BASE, 'Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Restormer = _mod.Restormer
def build_restormer():
    net = Restormer(inp_channels=3, out_channels=3, dim=48,
                    num_blocks=[4,6,6,8], num_refinement_blocks=4,
                    heads=[1,2,4,8], ffn_expansion_factor=2.66,
                    bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
    ckpt = torch.load(os.path.join(BASE, 'Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth'),
                      map_location='cpu')
    net.load_state_dict(ckpt['params'], strict=True)
    return net

test_model('Restormer', build_restormer, lambda net, x: net(x))
