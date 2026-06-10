from huggingface_hub import snapshot_download
import os

dest = os.path.join(os.path.dirname(__file__), '..', '..', 'PASD', 'runs', 'pasd_light')
os.makedirs(dest, exist_ok=True)

print(f'Downloading PASD light to {dest} ...')
snapshot_download(
    repo_id='yangtao9009/PASD',
    allow_patterns='pasd_light/**',
    local_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'PASD', 'runs'),
)
print('Done')
