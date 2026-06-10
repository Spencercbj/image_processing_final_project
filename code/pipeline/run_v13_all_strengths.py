"""
Run pipeline v13 for all three strengths with sharpen(0.5).
- s=0.8: add sharpen + re-run ESRGAN on existing DiffBIR output
- s=0.9, s=0.95: full DiffBIR → sharpen → ESRGAN (reuse step1/step2)
"""
import subprocess
import sys
import os

PYTHON = r'C:\Users\user\miniconda3\envs\deblur\python.exe'
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_pipeline_v13.py')

IMAGES = '01,05,08,11,12,15'
SHARED = 'pipeline_v13'

# First: add sharpen + ESRGAN to existing s=0.8 results
print('#' * 60)
print('  s=0.8: adding Sharpen(0.5) + ESRGAN x4')
print('#' * 60 + '\n')

cmd_08 = [
    PYTHON, SCRIPT,
    '--images', IMAGES,
    '--strength', '0.8',
    '--name', 'pipeline_v13',
    '--start_from', '4',
]
result = subprocess.run(cmd_08, stdout=sys.stdout, stderr=sys.stderr)
if result.returncode != 0:
    print(f'\nFAILED: s=0.8 sharpen+ESRGAN (exit code {result.returncode})')
    sys.exit(1)

# Then: s=0.9 and s=0.95
for strength in [0.9, 0.95]:
    s_tag = str(strength).replace('.', '')
    name = f'pipeline_v13_s{s_tag}'
    print(f'\n{"#" * 60}')
    print(f'  Running strength={strength} → {name}')
    print(f'{"#" * 60}\n')

    cmd = [
        PYTHON, SCRIPT,
        '--images', IMAGES,
        '--strength', str(strength),
        '--name', name,
        '--shared_dir', SHARED,
        '--start_from', '3',
    ]

    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f'\nFAILED: strength={strength} (exit code {result.returncode})')
        sys.exit(1)

print('\n' + '=' * 60)
print('All strength variants done!')
print('Compare:')
print('  s=0.8:  results/pipeline_v13/step5_realesrgan/')
print('  s=0.9:  results/pipeline_v13_s09/step5_realesrgan/')
print('  s=0.95: results/pipeline_v13_s095/step5_realesrgan/')
print('=' * 60)
