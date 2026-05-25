#!/bin/bash
#SBATCH --job-name=filter_unlabelled
#SBATCH --output=model_training_guide_spider/logs/filter_%j.out
#SBATCH --error=model_training_guide_spider/logs/filter_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_a100_7c
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

echo "=== SLURM job $SLURM_JOB_ID ==="
date
mkdir -p model_training_guide_spider/logs
nvidia-smi || echo "No GPU available"

# -------- ENV --------
source ~/test_venv/bin/activate
echo "Activated: $VIRTUAL_ENV"

# -------- Paths anchored to submit dir --------
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Input (unlabelled) H5 and where to save the filtered result
H5_IN="$SUBMIT_DIR/../../../../project/lofarsw/Data/Dynspec/full_data/unlabelled_dataset.h5"
OUT_DIR="$SUBMIT_DIR/../../../../project/lofarsw/Data/Dynspec"

# Python script that does orientation + background mitigation (Ir)
SCRIPT="$SUBMIT_DIR/filter_unlabelled.py"

echo "H5_IN   = $H5_IN"
echo "OUT_DIR = $OUT_DIR"
echo "SCRIPT  = $SCRIPT"

# -------- Staging --------
mkdir -p "$TMPDIR/job_$SLURM_JOB_ID"
rsync --info=progress2 "$H5_IN" "$SCRIPT" "$TMPDIR/job_$SLURM_JOB_ID"/
cd "$TMPDIR/job_$SLURM_JOB_ID" || exit 1

# -------- Run --------
# NOTE: The script writes <stem>_oriented_ir.h5 in --outdir (here: current dir).
BASENAME="$(basename "$H5_IN")"
STEM="${BASENAME%.*}"
EXT="${BASENAME##*.}"
OUT_FILE="${STEM}_oriented_ir.${EXT}"

python -u filter_unlabelled.py \
  --h5 "$BASENAME" \
  --outdir . \
  --k 1.2 \
  --T_short_s 45.0 \
  --f_cut_MHz 40.0 \
  --ramp_df_MHz 3.0 \
  --z_protect 2.0 \
  --sub_fraction_cap 0.6 \
  --robust

# -------- Copy it back --------
mkdir -p "$OUT_DIR"
if [[ -f "$OUT_FILE" ]]; then
  rsync --info=progress2 "$OUT_FILE" "$OUT_DIR"/
  echo "Saved filtered dataset to: $OUT_DIR/$OUT_FILE"
else
  echo "ERROR: Expected output $OUT_FILE not found." >&2
  ls -lah
  exit 2
fi

date
echo "=== Job $SLURM_JOB_ID completed ==="
