#!/bin/bash
set -e

EXP_NAME="apple"

MODEL_DIR="outputs/$EXP_NAME/gainer/0"
SIM_DIR="$MODEL_DIR/$EXP_NAME"
MP4_NAME="$EXP_NAME"
ITER=29999

echo -e "\n--- Step 1: Running Taichi Elements physics simulation ---"
ITER_PADDED=$(printf "%09d" "$ITER")
IN_FILE="$MODEL_DIR/step-${ITER_PADDED}_means.ply"

python simulation/apple.py \
  --in-file "$IN_FILE" \
  --out-dir "$SIM_DIR"

echo -e "\n--- Step 2: Preparing files for rendering ---"
rm -rf "$MODEL_DIR/camera_path"
mv "$SIM_DIR" "$MODEL_DIR/camera_path"

echo -e "\n--- Step 3: Rendering final animation with gainer-render ---"
gainer-render dataset \
  --load-config "$MODEL_DIR/config.yml" \
  --output-path "renders/$EXP_NAME"

echo -e "\n--- Step 4: Creating video from rendered frames ---"
ffmpeg -framerate 30 -i "renders/$EXP_NAME/test/rgb/%05d.jpg" \
  -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p "$MP4_NAME.mp4"

echo -e "\n--- All Done! Video saved to $MP4_NAME.mp4 ---"
