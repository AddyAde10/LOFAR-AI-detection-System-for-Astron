#!/bin/bash
#SBATCH --job-name=yolo-multilabel-ssl
#SBATCH --partition=gpu_a100_22c
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=model_training_guide_spider/logs/%x_%j.out
#SBATCH --error=model_training_guide_spider/logs/%x_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

# -------------------- PATHS --------------------
# Labeled datasets
H5_LABELED_FIL="../../project/lofarsw/Data/Dynspec/labelled_dataset_7classes_filtered.h5"
H5_LABELED_UNF="../../project/lofarsw/Data/Dynspec/labelled_dataset_7classes.h5"

# Unlabeled filtered (already oriented)
H5_UNLABELED_FIL="../../project/lofarsw/Data/Dynspec/unlabelled_dataset_oriented_ir.h5"

# Splits anchored on UNFILTERED labeled H5 (train_ids, val_ids, test_ids)
SPLITS_NPZ="../../project/lofarsw/Data/Dynspec/splits_labelled7_unfiltered_seed42.npz"

# Synthetic H5s (GAN)
H5_SYN2="lofar-runs/cdcgan/(H5.1500 T2)job_28687538/out/synthetic_type2_1500.h5"
H5_SYN4="lofar-runs/cdcgan/(H5.1500 T4,14)job_28687537/out/synthetic_type4,14_1500.h5"
H5_SYN5="lofar-runs/cdcgan/(H5.1500 T5 lil ovrfit)job_28693045/out/synthetic_type15,35_1500.h5"

PY_SSL="pretrain_ssl.py"
PY_TRAIN="train_multilabel_ssl.py"

OUTBASE="${SCRATCH:-$PWD}/lofar-runs/yolo_multilabel_ssl"
OUTDIR="$OUTBASE/job_${SLURM_JOB_ID}/out"

mkdir -p "$OUTDIR"
mkdir -p "model_training_guide_spider/logs"

# -------------------- Environment --------------------
cd ~
source test_venv/bin/activate
cd "$SLURM_SUBMIT_DIR"

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

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
for f in "$H5_LABELED_FIL" "$H5_LABELED_UNF" "$H5_UNLABELED_FIL" "$SPLITS_NPZ" "$H5_SYN2" "$H5_SYN4" "$H5_SYN5"; do
  if [[ ! -f "$f" ]]; then
    echo "WARNING: Missing file: $f" >&2
  fi
done
for f in "$H5_LABELED_FIL" "$H5_LABELED_UNF" "$H5_UNLABELED_FIL" "$SPLITS_NPZ" "$H5_SYN2"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: REQUIRED file missing: $f" >&2
    exit 2
  fi
done
for f in "$PY_SSL" "$PY_TRAIN"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Script not found: $f" >&2
    exit 2
  fi
done

du -h "$H5_LABELED_FIL" || true
du -h "$H5_LABELED_UNF" || true
du -h "$H5_UNLABELED_FIL" || true
du -h "$H5_SYN2" || true

# -------------------- Stage to node-local SSD --------------------
TMPDIR=${TMPDIR:-/tmp}
echo "TMPDIR=$TMPDIR"
mkdir -p "$TMPDIR/run"
df -h "$TMPDIR" || true

echo "Copying data and scripts to node-local TMPDIR..."
time rsync -av --info=progress2 "$H5_LABELED_FIL"   "$TMPDIR/run/labeled_filtered.h5"
time rsync -av --info=progress2 "$H5_LABELED_UNF"   "$TMPDIR/run/labeled_unfiltered.h5"
time rsync -av --info=progress2 "$H5_UNLABELED_FIL" "$TMPDIR/run/unlabelled_filtered.h5"
time rsync -av --info=progress2 "$SPLITS_NPZ"       "$TMPDIR/run/splits_labelled7_unfiltered_seed42.npz"
time rsync -av --info=progress2 "$H5_SYN2"          "$TMPDIR/run/synthetic_type2_1500.h5"

if [[ -f "$H5_SYN4" ]]; then
  time rsync -av --info=progress2 "$H5_SYN4" "$TMPDIR/run/synthetic_type4,14_1500.h5"
fi
if [[ -f "$H5_SYN5" ]]; then
  time rsync -av --info=progress2 "$H5_SYN5" "$TMPDIR/run/synthetic_type15,35_1500.h5"
fi

time rsync -av "$PY_SSL"   "$TMPDIR/run/pretrain_ssl.py"
time rsync -av "$PY_TRAIN" "$TMPDIR/run/train_multilabel_ssl.py"
cd "$TMPDIR/run"

# -------------------- (1) SSL Pretraining --------------------
echo "Starting SSL pretraining on (train-only twins + unlabeled filtered + synthetic)..."
SYN_ARGS=()
if [[ -f "synthetic_type4,14_1500.h5" ]]; then
  SYN_ARGS+=("synthetic_type4,14_1500.h5")
fi
if [[ -f "synthetic_type15,35_1500.h5" ]]; then
  SYN_ARGS+=("synthetic_type15,35_1500.h5")
fi
SYN_ARGS=("synthetic_type2_1500.h5" "${SYN_ARGS[@]}")

python -u pretrain_ssl.py \
  --h5_labeled_filtered   "$TMPDIR/run/labeled_filtered.h5" \
  --h5_labeled_unfiltered "$TMPDIR/run/labeled_unfiltered.h5" \
  --h5_unlabeled_filtered "$TMPDIR/run/unlabelled_filtered.h5" \
  --splits_npz            "$TMPDIR/run/splits_labelled7_unfiltered_seed42.npz" \
  --synth_h5              "${SYN_ARGS[@]}" \
  --outdir                "$TMPDIR/run/ssl_out" \
  --epochs 50 \
  --batch 64 \
  --img 224 224 \
  --sobel \
  --num_workers ${SLURM_CPUS_PER_TASK:-8}

SSL_CKPT="$TMPDIR/run/ssl_out/encoder_ssl.pth"
if [[ ! -f "$SSL_CKPT" ]]; then
  echo "ERROR: SSL checkpoint not found at $SSL_CKPT" >&2
  exit 3
fi

# -------------------- (2) Fine-tune Multilabel Classifier --------------------
echo "Starting fine-tuning with SSL weights (filtered train + GAN all types, unfiltered val/test)..."
GAN_ARGS=("$TMPDIR/run/synthetic_type2_1500.h5")
if [[ -f "$TMPDIR/run/synthetic_type4,14_1500.h5" ]]; then
  GAN_ARGS+=("$TMPDIR/run/synthetic_type4,14_1500.h5")
fi
if [[ -f "$TMPDIR/run/synthetic_type15,35_1500.h5" ]]; then
  GAN_ARGS+=("$TMPDIR/run/synthetic_type15,35_1500.h5")
fi

python -u train_multilabel_ssl.py \
  --h5_labeled_filtered   "$TMPDIR/run/labeled_filtered.h5" \
  --h5_labeled_unfiltered "$TMPDIR/run/labeled_unfiltered.h5" \
  --splits_npz            "$TMPDIR/run/splits_labelled7_unfiltered_seed42.npz" \
  --gan_h5                "${GAN_ARGS[@]}" \
  --ssl_ckpt              "$TMPDIR/run/ssl_out/encoder_ssl.pth" \
  --outdir                "$TMPDIR/run/clf_out" \
  --labels 1 2 3 4 5 6 7 \
  --loss bce \
  --head_warmup 5 \
  --unfreeze_stages_at 4 12 \
  --lrd 0.1 \
  --epochs 50 \
  --batch 32 \
  --img 224 224 \
  --use_sampler \
  --sobel \
  --num_workers ${SLURM_CPUS_PER_TASK:-8}

# -------------------- Save results --------------------
echo "Copying results back to $OUTDIR..."
mkdir -p "$OUTDIR"
rsync -av "$TMPDIR/run/ssl_out"/ "$OUTDIR/ssl_out"/
rsync -av "$TMPDIR/run/clf_out"/ "$OUTDIR/clf_out"/

# -------------------- Print summary --------------------
echo "Per-class metrics (any depth):"
find "$OUTDIR/clf_out" -maxdepth 3 -name "per_class_metrics.csv" -print

echo "Test report(s):"
python - <<'PY'
import json, os, glob
base = os.path.join(os.environ["OUTDIR"], "clf_out")
paths = glob.glob(os.path.join(base, "**", "test_report.json"), recursive=True)
if not paths:
    print("test_report.json not found")
else:
    for p in paths:
        print("==>", p)
        with open(p) as f:
            print(json.dumps(json.load(f), indent=2))
PY

echo "✅ DONE. Outputs in: $OUTDIR"
