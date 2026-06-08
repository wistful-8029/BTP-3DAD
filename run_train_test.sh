#!/usr/bin/env bash
set -euo pipefail

CLASSES=(car chicken gemstone)
SEED=111

DATA_ROOT="${DATA_ROOT:-./data/Real3D-AD-2048-npz}"
MODEL_PATH="${MODEL_PATH:-./pretrained/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/train_test_seed111}"
TRAIN_ROOT="${TRAIN_ROOT:-${OUTPUT_ROOT}/train}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/eval_standard_aupro}"

mkdir -p "${TRAIN_ROOT}" "${EVAL_ROOT}"

for class_name in "${CLASSES[@]}"; do
  run_idx=0
  seed="${SEED}"
  run_dir="${TRAIN_ROOT}/Real3D/${class_name}/run_00_seed${seed}"
  ckpt="${run_dir}/best.pth"
  eval_dir="${EVAL_ROOT}/${class_name}/run_00_seed${seed}"

  echo "================================================================"
  echo "Train Real3D ${class_name} | run=${run_idx} | seed=${seed}"
  echo "================================================================"
  if [[ -f "${ckpt}" ]]; then
    echo "Found existing checkpoint, skip training: ${ckpt}"
  else
    python train.py \
      --dataset_name Real3D \
      --data_root "${DATA_ROOT}" \
      --model_path "${MODEL_PATH}" \
      --output_root "${TRAIN_ROOT}" \
      --train_class "${class_name}" \
      --train_split test \
      --run_idx "${run_idx}" \
      --seed "${seed}" \
      ${EXTRA_TRAIN_ARGS:-}
  fi

  echo "================================================================"
  echo "Test Real3D ${class_name} | run=${run_idx} | seed=${seed}"
  echo "================================================================"
  python test_standard_aupro.py \
    --dataset_name Real3D \
    --data_root "${DATA_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --ckpt_root "${ckpt}" \
    --train_class "${class_name}" \
    --output_dir "${eval_dir}" \
    --save_dir "${eval_dir}/visual" \
    --fullres_eval \
    ${EXTRA_TEST_ARGS:-}
done

if [[ -f "scripts/aggregate_real3d_runs.py" ]]; then
  python scripts/aggregate_real3d_runs.py \
    --eval_root "${EVAL_ROOT}" \
    --out_dir "${EVAL_ROOT}"
fi

echo "Done."
echo "Per-run summary: ${EVAL_ROOT}/summary_by_run.csv"
echo "Mean/std summary: ${EVAL_ROOT}/summary_mean_std.csv"
