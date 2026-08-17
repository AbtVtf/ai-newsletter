#!/usr/bin/env bash
# Launcher for prop_gen.py inside the trellis2 conda env, mirroring how the
# MCP server starts. The Blender addon calls this, not the MCP server: that
# one speaks stdio and is owned by whichever client spawned it.
set -e
source /home/mafuu/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python /home/mafuu/Documents/GitHub/ai-newsletter/video/prop_gen.py "$@"
