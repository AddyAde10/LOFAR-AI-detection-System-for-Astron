#!/bin/bash
#SBATCH --job-name=yolo-multilabel
#SBATCH --partition=gpu_a100_22c          # or gpu_a100_7c
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=model_training_guide_spider/logs/%x_%j.out
#SBATCH --error=model_training_guide_spider/logs/%x_%j.err
# SBATCH --account=<your_account>

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

# -------------------- EDIT THESE --------------------
# Filtered labelled dataset
H5_FILTERED="../../project/lofarsw/Data/Dynspec/labelled_dataset_7classes_filtered.h5"

# Optional GAN H5s for types 2/4/5 (paths on shared FS)
GAN_H5_1="lofar-runs/cdcgan/(H5.1500 T2)job_28687538/out/synthetic_type2_1500.h5"
GAN_H5_2="lofar-runs/cdcgan/(H5.1500 T4,14)job_28687537/out/synthetic_type4,14_1500.h5"
GAN_H5_3="lofar-runs/cdcgan/(H5.1500 T5 lil ovrfit)job_28693045/out/synthetic_type15,35_1500.h5"

PY_SCRIPT="train_multilabel.py"

# Persistent splits file (on shared filesystem, not TMP)
SPLIT_PATH="../../project/lofarsw/Data/Dynspec/splits_yolo_labelled7_filtered_seed42.npz"

OUTBASE="${SCRATCH:-$PWD}/lofar-runs/yolo_multilabel"
OUTDIR="$OUTBASE/job_${SLURM_JOB_ID}/out"

# -------------------- Prep dirs --------------------
mkdir -p "$OUTDIR"
mkdir -p "model_training_guide_spider/logs"

# -------------------- Environment --------------------
cd ~
source test_venv/bin/activate
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# -------------------- Sanity prints --------------------
python - <<'PY'
import torch, os, pathlib
print(f"=== SLURM job {os.environ.get('SLURM_JOB_ID')} ===")
print("Submit dir:", pathlib.Path.cwd())
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

# -------------------- Validate inputs --------------------
for f in "$H5_FILTERED"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: H5 not found: $f" >&2
    ls -lah "$(dirname "$f")" || true
    exit 2
  fi
done

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: Script not found: $PY_SCRIPT" >&2
  ls -lah .
  exit 2
fi

du -h "$H5_FILTERED" || true

# -------------------- Stage to node-local SSD --------------------
TMPDIR=${TMPDIR:-/tmp}
echo "TMPDIR=$TMPDIR"
mkdir -p "$TMPDIR/run"
df -h "$TMPDIR" || true

echo "Copying data and script to node-local TMPDIR..."
time rsync -av --info=progress2 "$H5_FILTERED" "$TMPDIR/run/filtered.h5"

# GAN files are optional: only copy if they exist
GAN_ARGS=()
for g in "$GAN_H5_1" "$GAN_H5_2" "$GAN_H5_3"; do
  if [[ -f "$g" ]]; then
    base=$(basename "$g")
    time rsync -av --info=progress2 "$g" "$TMPDIR/run/$base"
    GAN_ARGS+=("$TMPDIR/run/$base")
  fi
done

time rsync -av "$PY_SCRIPT" "$TMPDIR/run/train_multilabel.py"
cd "$TMPDIR/run"

# -------------------- Train --------------------
echo "Starting YOLO multilabel training (filtered REAL + GAN on TRAIN only; VAL/TEST = REAL filtered)..."
python -u train_multilabel.py \
  --h5 "$TMPDIR/run/filtered.h5" \
  ${GAN_ARGS:+--h5_gan "${GAN_ARGS[@]}"} \
  --outdir "$TMPDIR/run/out" \
  --labels 1 2 3 4 5 6 7 \
  --img 224 224 \
  --batch 32 \
  --epochs 120 \
  --lr 7.14e-4 \
  --wd 1e-4 \
  --val_split 0.15 \
  --test_split 0.15 \
  --num_workers ${SLURM_CPUS_PER_TASK:-8} \
  --class_sampler balanced \
  --loss focal --focal_gamma 2.0 --focal_alpha 0.25 \
  --es_patience 12 --es_min_delta 0.0 \
  --threshold_mode perclass \
  --split_path "$SPLIT_PATH"

# -------------------- Save results --------------------
echo "Copying results back to $OUTDIR..."
mkdir -p "$OUTDIR"
rsync -av "$TMPDIR/run/out"/ "$OUTDIR"/

echo "=== Saved Artifacts ==="
ls -lah "$OUTDIR" || true
echo "Per-class metrics:"
find "$OUTDIR" -maxdepth 1 -type f -name "per_class_metrics.*" -print || true
echo "Confusion matrices:"
find "$OUTDIR" -maxdepth 1 -type f -name "confusion_matrices.*" -print || true
echo "Per-class thresholds:"
find "$OUTDIR" -maxdepth 1 -type f -name "per_class_thresholds.json" -print || true

echo "Test report:"
python - <<'PY'
import json, os
base = os.environ.get("OUTDIR")
p = os.path.join(base, "test_report.json")
print(json.dumps(json.load(open(p)), indent=2) if os.path.isfile(p) else "test_report.json not found")
PY

echo "✅ DONE. Outputs in: $OUTDIR"
