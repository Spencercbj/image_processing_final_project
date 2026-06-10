#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ACTION="${1:-help}"
shift || true

PIPELINE="code/pipeline/run_darkir_restormer_diffbir_v2.py"
FULL_OUT="${FULL_OUT:-results/pipeline_darkir_restormer_diffbir_v2}"
CANDIDATE_OUT="${CANDIDATE_OUT:-results/pipeline_darkir_restormer_diffbir_v2_candidates}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/colab_run_darkir_restormer_diffbir_v2.sh <action>

Actions:
  check              Verify required files and compile the pipeline.
  smoke08            Run image 08 without ESRGAN.
  smoke13            Run image 13 through DarkIR without DiffBIR/ESRGAN.
  candidates         Run 07,10,13,14,15 with the full default pipeline.
  full               Run all 15 images with the full default pipeline.
  resume <step>      Resume full output from a pipeline step number.
  ablation_darkir    Test DarkIR on 03,06,13,14,15 without ESRGAN.
  ablation_diffbir   Test DiffBIR strength 0.7 on 01,06,10,14,15 without ESRGAN.
  ablation_no_omdnet Test 07,14 without OMDNet/ESRGAN.
  package_full       Zip full output and copy it to Drive.
  package_experiments Zip smoke/candidate/ablation outputs and copy them to Drive.
  custom --args ...  Pass custom args directly to the Python pipeline.

Environment overrides:
  FULL_OUT=results/my_full
  CANDIDATE_OUT=results/my_candidates
  DRIVE_OUT=/content/drive/MyDrive
USAGE
}

check_required() {
  python - <<'PY'
from pathlib import Path

required = [
    'code/pipeline/run_darkir_restormer_diffbir_v2.py',
    'DarkIR/models/DarkIR_384.pt',
    'Restormer/Motion_Deblurring/pretrained_models/motion_deblurring.pth',
    'OMDNet/checkpoints/test1/model_ckpt_epoch_219.ckpt',
    'Real-ESRGAN/weights/RealESRGAN_x4plus.pth',
    'DiffBIR/inference.py',
]

missing = [p for p in required if not Path(p).exists()]
if missing:
    print('Missing required files:')
    for p in missing:
        print('  -', p)
    raise SystemExit(1)

print('required files present')
PY
  python -m py_compile "$PIPELINE"
}

run_pipeline() {
  python "$PIPELINE" "$@"
}

package_to_drive() {
  local zip_name="$1"
  shift
  local drive_out="${DRIVE_OUT:-/content/drive/MyDrive}"
  zip -r "$zip_name" "$@"
  if [[ -d "$drive_out" ]]; then
    cp "$zip_name" "$drive_out/"
  else
    echo "WARNING: DRIVE_OUT not found: $drive_out"
  fi
  ls -lh "$zip_name"
}

case "$ACTION" in
  check)
    check_required
    ;;
  smoke08)
    run_pipeline \
      --images 08 \
      --skip_esrgan \
      --out_dir results/smoke_darkir_restormer_diffbir_v2_08
    find results/smoke_darkir_restormer_diffbir_v2_08 -maxdepth 2 -type f | sort
    ;;
  smoke13)
    run_pipeline \
      --images 13 \
      --darkir_ids 13 \
      --no_diffbir 13 \
      --skip_esrgan \
      --out_dir results/smoke_darkir_v2_13
    find results/smoke_darkir_v2_13 -maxdepth 2 -type f | sort
    ;;
  candidates)
    run_pipeline \
      --images 07,10,13,14,15 \
      --out_dir "$CANDIDATE_OUT"
    ;;
  full)
    run_pipeline \
      --out_dir "$FULL_OUT"
    ;;
  resume)
    step="${1:-6}"
    run_pipeline \
      --start_from "$step" \
      --out_dir "$FULL_OUT"
    ;;
  ablation_darkir)
    run_pipeline \
      --images 03,06,13,14,15 \
      --darkir_ids 03,06,13,14,15 \
      --skip_esrgan \
      --out_dir results/ablation_darkir_more_ids
    ;;
  ablation_diffbir)
    run_pipeline \
      --images 01,06,10,14,15 \
      --diffbir_strength 0.7 \
      --skip_esrgan \
      --out_dir results/ablation_diffbir_s07
    ;;
  ablation_no_omdnet)
    run_pipeline \
      --images 07,14 \
      --skip_omdnet \
      --skip_esrgan \
      --out_dir results/ablation_no_omdnet
    ;;
  package_full)
    package_to_drive pipeline_darkir_restormer_diffbir_v2.zip "$FULL_OUT"
    ;;
  package_experiments)
    package_to_drive darkir_v2_experiments.zip \
      results/smoke_darkir_restormer_diffbir_v2_08 \
      results/smoke_darkir_v2_13 \
      "$CANDIDATE_OUT" \
      results/ablation_darkir_more_ids \
      results/ablation_diffbir_s07 \
      results/ablation_no_omdnet
    ;;
  custom)
    run_pipeline "$@"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown action: $ACTION"
    usage
    exit 2
    ;;
esac
