#!/bin/bash
#SBATCH --job-name=orient_unf
#SBATCH --output=model_training_guide_spider/logs/orient_unf_%j.out
#SBATCH --error=model_training_guide_spider/logs/orient_unf_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100_22c          # or gpu_a100_22c
#SBATCH --gpus=a100:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

echo "=== SLURM job $SLURM_JOB_ID ==="
date
mkdir -p model_training_guide_spider/logs
nvidia-smi || echo "No GPU available (not required)"

# Env
export HDF5_USE_FILE_LOCKING=FALSE      # avoids locking quirks on shared FS
source ~/test_venv/bin/activate
echo "Activated: $VIRTUAL_ENV"

# Paths anchored to submit dir
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
H5_IN="$SUBMIT_DIR/../../../../project/lofarsw/Data/Dynspec/unlabelled_dataset.h5"
OUT_DIR="$SUBMIT_DIR/../../../../project/lofarsw/Data/Dynspec/"
SCRIPT="$SUBMIT_DIR/orient_unlabelled_unf.py"

echo "H5_IN   = $H5_IN"
echo "OUT_DIR = $OUT_DIR"
echo "SCRIPT  = $SCRIPT"

# Stage only the script (NOT the 40+ GB H5)
mkdir -p "$TMPDIR/job_$SLURM_JOB_ID"
rsync --info=progress2 "$SCRIPT" "$TMPDIR/job_$SLURM_JOB_ID"/
cd "$TMPDIR/job_$SLURM_JOB_ID"

# Run: read H5 directly from source path, write output in TMPDIR
BASENAME="$(basename "$H5_IN")"
STEM="${BASENAME%.*}"
EXT="${BASENAME##*.}"
OUT_FILE="${STEM}_oriented_unf.${EXT}"

python -u "$(basename "$SCRIPT")" \
  --h5 "$H5_IN" \
  --outdir . \
  --compression lzf

# Copy result back
mkdir -p "$OUT_DIR"
rsync --info=progress2 --inplace "$OUT_FILE" "$OUT_DIR"/
echo "Saved oriented (unfiltered) dataset to: $OUT_DIR/$OUT_FILE"

date
echo "=== Job $SLURM_JOB_ID completed ==="
