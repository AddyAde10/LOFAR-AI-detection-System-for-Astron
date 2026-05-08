#!/bin/bash
#SBATCH --job-name=cdcgan_srb_t2
#SBATCH --partition=gpu_a100_22c      # Using your partition
#SBATCH --gpus=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00             # Increased to 2 days
#SBATCH --output=model_training_guide_spider/logs/%x_%j.out
#SBATCH --error=model_training_guide_spider/logs/%x_%j.err

set -euo pipefail

# ---- User paths (edit these) ----
# !! EDIT THIS to point to your unfiltered data !!
UNFILTERED_H5_PATH="../../project/lofarsw/Data/Dynspec/labelled_dataset_7classes.h5"

H5_PATH="../../project/lofarsw/Data/Dynspec/labelled_dataset_7classes_filtered.h5"
OUTROOT="${SCRATCH:-$PWD}/lofar-runs/cdcgan"
OUTDIR="$OUTROOT/job_${SLURM_JOB_ID}/out"
SCRIPT_LOCAL="train_cdcgan_srb.py" # <-- Updated script name

# ---- ENV ----
module purge || true
source ~/test_venv/bin/activate # <-- Using spider venv path

echo "CUDA available: $(python - <<'PY'
import torch; print(torch.cuda.is_available())
PY
)"

# ---- Setup Dirs ----
mkdir -p "$OUTDIR" "model_training_guide_spider/logs"
mkdir -p "$TMPDIR/job_$SLURM_JOB_ID"
cd "$TMPDIR/job_$SLURM_JOB_ID"

# ---- Stage script & paths ----
H5_LOCAL="$H5_PATH"  # read in place
UNFILTERED_H5_LOCAL="$UNFILTERED_H5_PATH" # read in place

# Copy the new script to the compute node
rsync --info=progress2 "$SLURM_SUBMIT_DIR/train_cdcgan_srb.py" "./$SCRIPT_LOCAL"

set -x
# ---- Execute the new training script ----
python -u "$SCRIPT_LOCAL" \
  --h5 "$H5_LOCAL" \
  --unfiltered_h5_path "$UNFILTERED_H5_LOCAL" \
  --outdir "$OUTDIR" \
  --target_labels "15,35" \
  --epochs 400 \
  --batch 16 \
  --imgH 256 --imgW 160 \
  --nz 128 --ngf 64 --ndf 64 \
  --lr 1e-4 --lrD 8e-5 --lrG 2e-4 --betas 0.5 0.999 \
  --loss "hinge" --r1_gamma 40.0 --gp_lambda 10.0 \
  --use_ema --ema_decay 0.999 \
  --strong_aug \
  --balanced_batch \
  --save_every 50 \
  --aug_target_count 1000 \
  --num_workers $SLURM_CPUS_PER_TASK \
  --num_gen 250 \
  --g_steps 2 \
  --fid_num_samples 1000 \
  --num_h5 1500 \
  --h5_outname "synthetic_type15,35_1500.h5"

set +x

# ---- Sync results back ----
# This is crucial for copying data from $TMPDIR back to $SCRATCH
mkdir -p "$OUTDIR"
rsync -av --info=progress2 ./ "$OUTDIR"/

echo "Done. Artifacts in: $OUTDIR"