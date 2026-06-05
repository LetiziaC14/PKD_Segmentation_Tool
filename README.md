FloodFill interactive segmentation
=================================

Overview
--------
This repository contains a lightweight, standalone Python tool for interactive segmentation of medical images. It combines a seed-based intensity selection with a spatial radius constraint and post-processing to produce solid (hole-free) object masks. The workflow is meant for quick manual annotation and review.

Algorithm
---------
- Seed-based selection: on left-click the tool reads the grayscale intensity at the seed and selects pixels whose intensity is within `tol` of the seed value and within `radius` pixels from the seed.
- Connected component: only the connected component containing the seed is kept (so distant similar-intensity regions are excluded).
- Hole-filling: masks are post-processed (morphological closing and flood-fill) to remove internal holes so the segmentation is a single filled surface.

Features
--------
- Interactive seed-based segmentation with adjustable `tol` and `radius`.
- Manual painting/erasing (`m` to toggle); `brush` trackbar controls brush size.
- Undo (`u`), reset (`r`), create empty mask (`c`).
- Save outputs including an RGB class mask and overlay preview.

Requirements
------------
- Python 3.8+
- OpenCV and NumPy

Install
-------
```
pip install -r requirements.txt
```

Quick start
-----------
```
python floodfill_segmentation.py -i /path/to/images -o masks -t 10
```

Controls
--------
- Left-click: add seed and create mask
- `tol` (trackbar): intensity tolerance
- `radius` (trackbar): spatial radius in pixels
- `m`: toggle manual paint mode (left add, right erase)
- `brush` (trackbar): brush size
- `zoom` (trackbar): display zoom factor for precise seeding
- `u`: undo last mask
- `r`: reset masks
- `c`: create empty mask
- `s`: save masks
- `v`: cycle display (`both`, `overlay`, `original`)
- `n` / `p`: next / previous image
- `q`: quit

Output
------
Saved files are placed in `INPUT_FOLDER/OUTPUT_FOLDER` (default `INPUT_FOLDER/masks`):
- `image_basename_masks_overlay.png` — overlay preview of masks
- `image_basename_classmask.png` — RGB image where each mask is colored distinctly
- optionally, individual binary masks `image_basename_mask_1.png`, ... (use `--no-binaries` to skip them)

Example
-------
An example output image is included: `example_output.png` (class-colored masks on a sample image).

Inline example overlay for your dataset:

![Overlay example](A/20240916-RM%20ABDOMEN(FP)/6001-EX%20AX%20T2%20FS/masks/6001-EX_AX_T2_FS_000000_masks_overlay.png)
