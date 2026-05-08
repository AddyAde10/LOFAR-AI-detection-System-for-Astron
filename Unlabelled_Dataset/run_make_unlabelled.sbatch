#!/bin/bash
#SBATCH --job-name=make_unlabelled
#SBATCH --output=model_training_guide_spider/logs/make_unlabelled_%j.out
#SBATCH --error=model_training_guide_spider/logs/make_unlabelled_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100_7c
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

echo "=== SLURM job $SLURM_JOB_ID ==="
date
nvidia-smi || echo "No GPU available"

# -------- ENV --------
source ~/test_venv/bin/activate
echo "Activated: $VIRTUAL_ENV"

# -------- Paths anchored to submit dir --------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
FULL_H5="$SUBMIT_DIR/../../../../project/lofarsw/Data/Dynspec/full_data/dset_2025.h5"
LABELLED_H5="$SUBMIT_DIR/../../../../project/lofarsw/Data/Dynspec/labelled_dataset_7classes.h5"
OUT_DIR="$SUBMIT_DIR/../../../../project/lofarsw/Data/Dynspec/full_data"
SCRIPT="$SUBMIT_DIR/make_unlabelled_from_timestamps.py"

echo "FULL_H5     = $FULL_H5"
echo "LABELLED_H5 = $LABELLED_H5"
echo "OUT_DIR     = $OUT_DIR"
echo "SCRIPT      = $SCRIPT"

# -------- Staging --------
mkdir -p "$TMPDIR/job_$SLURM_JOB_ID"
rsync --info=progress2 "$FULL_H5" "$LABELLED_H5" "$SCRIPT" "$TMPDIR/job_$SLURM_JOB_ID"/
cd "$TMPDIR/job_$SLURM_JOB_ID" || exit 1

# -------- Run --------
python -u make_unlabelled_from_timestamps.py \
  --full dset_2025.h5 \
  --labelled labelled_dataset_7classes.h5
  --compression lzf

# Copy it back
mkdir -p "$OUT_DIR"
rsync --info=progress2 unlabelled_dataset.h5 "$OUT_DIR"/

echo "Saved unlabelled dataset to: $OUT_DIR/unlabelled_dataset.h5"
date
echo "=== Job $SLURM_JOB_ID completed ==="
