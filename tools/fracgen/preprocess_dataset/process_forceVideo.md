
## PhysGaussian dataset processing

### Force-only (legacy)

Use `data_processed/process_forceVideo.py` to build a dataset with:
- **RGB video**: from numbered `*.png` frames
- **force_video**: from `2d_force_map/*.npy`

Output:
- `data/<dst>/videos/*.mp4`
- `data/<dst>/force_videos/*.mp4`
- `data/<dst>/metadata.csv` with columns `video,prompt,force_video`

### Multi-modal (strain/stress/vel/damage + RGB)

Use `data_processed/process_multimodalVideo.py` to build a dataset with:
- **video**: RGB mp4 (from `RGB_video/*.png` if present, otherwise from numbered `*.png` in the sample folder)
- **strain_video**: from `2d_strain_map/*.npy`
- **stress_video**: from `2d_stress_map/*.npy`
- **flow_video**: from `2d_vel_map/*.npy`
- **damage_video**: from `2d_damage_map/*.npy`
- **force_video** (optional): from `2d_force_map/*.npy` if the folder exists

Example:

```bash
python data_processed/process_multimodalVideo.py \
  --src_root /weka/scratch/bhattad/tnguy303/research/PhysGaussian/output_exp_update_gs \
  --dst_root data/physgaussian_multimodal_update_gs \
  --num_frames 81 \
  --fps 25
```

This creates:
- `data/physgaussian_multimodal_update_gs/videos/`
- `data/physgaussian_multimodal_update_gs/{strain,stress,flow,damage,force}_videos/`
- `data/physgaussian_multimodal_update_gs/metadata.csv` with columns:
  `video,prompt,force_video,stress_video,flow_video,damage_video,strain_video`


## how generate the visualized videos?

The `2d_*_map/*.npy` files are converted to mp4 by:
- loading each `0000.npy`, `0001.npy`, ... (aligned by frame index with RGB)
- converting the numpy array into an **RGB image** $(H \times W \times 3)$
- scaling each channel into $[0, 1]$
- turning it into a **uint8 RGB** frame in $[0,255]$
- writing frames sequentially into an `.mp4` stream

### 1) What shapes (channel dims) are supported?

For each `*.npy` map frame we support these common layouts:
- **(H, W)**: treated as 1-channel and then mapped to RGB by replication
- **(C, H, W)** where $C \in \{1,2,3\}$: channel-first → transposed to (H, W, C)
- **(H, W, C)** where $C \in \{1,2,3\}$: channel-last → used as-is

If the array has **more than 3 channels** (e.g. (4,H,W) or (H,W,4)), we **raise an error** (we do not drop channels).

If the array has a different shape, the script will raise an error so we don’t silently write wrong videos.

### 2) How do we convert to 3 RGB channels?

We first ensure the map is interpreted as **up to 3 channels** (C=1/2/3), then map to exactly 3 channels:
- **C=1**: replicate → (R,G,B) = (x, x, x)
- **C=2**: pad a zero channel → (R,G,B) = (x0, x1, 0)
- **C=3**: direct mapping → (R,G,B) = (x0, x1, x2)

After that, each channel is normalized into $[0,1]$, then encoded as:
- `u8 = (rgb * 255 + 0.5).astype(uint8)` → shape **(H, W, 3)**

So the output map videos are always true RGB frames (sometimes “grayscale RGB” if C=1, and “RG+0” if C=2), which matches the video loader expecting RGB frames.

### 2.1) Special case: `flow_video` (2d_vel_map) visualization

For `flow_video` we use a more standard optical-flow visualization instead of directly mapping channels to RGB:
- Input must be a **2-channel vector field** (u,v) in either (H,W,2) or (2,H,W).
- We convert each pixel to HSV where:
  - **Hue** = direction \( \theta = \mathrm{atan2}(v,u) \) (angle mapped to \([0,1]\))
  - **Saturation** = normalized magnitude \( \| (u,v) \| / r_{\max} \)
  - **Value** = 1
- Then we convert HSV → RGB and write that as the flow frame.

Magnitude scale \(r_{\max}\) is chosen per video by default using a robust percentile:
- `--flow_norm percentile --flow_percentile 99 --flow_sample_stride 8`

You can switch to max-magnitude scaling with `--flow_norm max`.

### 3) How are values normalized into $([0,1])$?

Controlled by `--normalize` in `process_multimodalVideo.py`:
- **`minmax_per_video` (default)**: compute global $v_\min, v_\max$ over all frames in that modality video, then
$$
x'=\mathrm{clip}\left(\frac{x-v_\min}{v_\max-v_\min}, 0, 1\right)
$$
- **`minmax_per_frame`**: same formula but $v_\min, v_\max$ computed per-frame (can cause temporal flicker)
- **`clip01`**: directly clamp to $[0,1]$ (useful if the source is already normalized)

You can also multiply raw values before normalization via `--map_gain`.

### 4) Optional post-transform (contrast shaping)

After normalization you can apply:
- **`--transform none`**: no change
- **`--transform gamma`**: $y = x^{\gamma}$ (use `--gamma`)
- **`--transform log`**: $y = \log(1+\alpha x)/\log(1+\alpha)$ (use `--log_alpha`)


