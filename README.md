# Habitat-Scene-Speedup

Faster rendering of Matterport3D (MP3D) and HM3D scenes in [Habitat](https://github.com/facebookresearch/habitat-sim),
by merging mesh primitives that share a material. The merged `.glb` is a drop-in replacement for the original file:
same triangles, same materials, same textures, and in almost every case the same pixels.

On MP3D, rendering an RGB + depth view takes 55% less time on average, and a 12-view panorama in the two heaviest
scenes drops from 170–250 ms to 7–11 ms. On HM3D the saving is smaller, about 18%, because its scenes are split into
fewer pieces to begin with.

## Why

Habitat-sim issues one draw call per glTF primitive, and rendering these scans is limited by the CPU cost of issuing
draw calls rather than by the number of triangles. The MP3D files split each house into many small primitives (at most
65,535 vertices each, since they use 16-bit indices), up to 7,490 in one scene. Render time grows linearly with the
primitive count: about 0.08 ms + 1.26 ms per 500 primitives for one RGB 224×224 + depth 256×256 view (habitat-sim
0.1.7, AMD EPYC 9354).

The two heaviest MP3D scenes render 33–37× slower than the median one. Training loops usually run many episodes in a
row in the same scene, so throughput drops sharply whenever environments land on them.

`merge_glb.py` concatenates the vertex and index buffers of all primitives that use the same material into one
primitive. Vertex data, triangles, materials and texture bytes are copied unchanged.

## Results

| | MP3D (90 scenes) | HM3D v0.1 (900 scenes) |
|---|---|---|
| Primitives | 44,404 → 9,552 | 156,836 → 45,137 |
| Median primitives per scene | 245 → 80 | 146.5 → 43 |
| RGB + depth per view, mean over scenes, habitat-sim 0.1.7 | 1.14 → 0.52 ms | – |
| RGB + depth per view, mean over scenes, habitat-sim 0.3.0 | 0.87 → 0.39 ms | 0.33 → 0.27 ms (val, 100 scenes) |
| Heaviest scene, 12-view panorama | 251 → 7.6 ms (0.1.7), 183 → 11 ms (0.3.0) | – |
| Scenes rendered pixel-identical | 87 / 90 | 100 / 100 (val) |
| Rebuilt navmesh identical (agent 0.88 m / 0.18 m) | 89 / 90 | 100 / 100 (val) |

"Per view" means one RGB 224×224 image plus one depth 256×256 image, each drawn and read back to the CPU. Each scene is
rendered from 50 random navigable poses. The habitat-sim 0.1.7 numbers were measured on an AMD EPYC 9354 with an
RTX 4090, the 0.3.0 numbers on an Intel i9-14900K with an RTX 4090. Per-scene numbers are in [`results/`](results).

Weighted by how often each scene appears in training, the per-view render time on MP3D drops by 66% for R2R-CE and 72%
for RxR-CE (habitat-sim 0.1.7). Most of the gain comes from a few heavy scenes; the median scene improves by 11–21%.
After merging, the primitive count accounts for only about 10% of the remaining render time, which is mostly
per-camera overhead.

Gibson scenes are already a single primitive each and do not need this.

## Usage

Requires Python 3.6+ and numpy.

```bash
# one file
python merge_glb.py in.glb out.glb

# a whole dataset in place; each original is kept as <name>.orig
python merge_glb.py --scenes-dir data/scene_datasets/mp3d
python merge_glb.py --scenes-dir data/scene_datasets/hm3d/train --pattern '*/*.basis.glb'

# put the originals back
python merge_glb.py --scenes-dir data/scene_datasets/mp3d --restore
```

Running it again is safe: it always rebuilds from `<name>.orig`. `.navmesh`, `.house` and `_semantic.ply` files are
not touched. Processing all 90 MP3D scenes takes about 10 seconds.

[`results/mp3d_md5.txt`](results/mp3d_md5.txt) and [`results/hm3d_v0.1_md5.txt`](results/hm3d_v0.1_md5.txt) list the
md5 of every original and merged file, so you can check that your inputs and outputs match ours.

## How close is the output to the original?

**Geometry.** In every scene, each material's triangle list (all vertex attributes, in order) is byte-identical to the
original. `tools/check_geometry.py` verifies this without a GPU.

**Rendering.** Habitat draws primitives in file order. In 80 of the 90 MP3D scenes and in all HM3D scenes, the original
file already lists its primitives grouped by material. The merged file then contains the same triangles in the same
order, and the images are identical by construction.

In the other 10 MP3D scenes (`aayBHfsNo7d b8cTxDM8gDG D7G3Y4RVNrH fzynW3qQPVF jtcxE69GiFV kEZ7cmS4wCh V2XKFyX4ASd
VVfe2KiqLaN XcA2TqTSSAj yqstnuAEVhm`), materials are interleaved. Merging changes which of two coplanar, overlapping
faces with different materials is drawn first. We saw this in three of them (`fzynW3qQPVF`, `V2XKFyX4ASd`,
`kEZ7cmS4wCh`): at most 9 of 2.5 million pixels differ in 224×224 RGB, and at most 39 of 15.4 million at 640×480.
Depth is identical.

**Navmesh.** When habitat rebuilds the navmesh from the scene mesh, the result depends on triangle order. habitat-sim
0.1.7 does this automatically when the agent height or radius differs from the default, for example in the RxR-CE
setting (0.88 m / 0.18 m). For MP3D, 89 of 90 rebuilt navmeshes are identical. In `kEZ7cmS4wCh` the navigable area
grows from 680.42 to 680.53 m², while the geodesic distances and the navigability of all RxR-CE reference paths in that
scene stay the same.

None of the 10 interleaved scenes appear in the val_unseen splits of R2R-CE or RxR-CE.

## Checking it yourself

```bash
# geometry, no GPU needed (about 20 s for MP3D)
python tools/check_geometry.py --scenes-dir data/scene_datasets/mp3d

# render both versions from the same poses, compare pixels, time each camera
python tools/check_render.py --scenes-dir data/scene_datasets/mp3d

# rebuild the navmesh from both versions and compare
python tools/check_navmesh.py --scenes-dir data/scene_datasets/mp3d --height 0.88 --radius 0.18
```

All three take `original.glb merged.glb` instead of `--scenes-dir` to check a single pair, and `--pattern
'*/*.basis.glb'` for HM3D. The render and navmesh checks run on habitat-sim 0.1.7 and 0.3.0.

## Supported input

The script accepts glTF binary files in which all mesh nodes share one transform and all primitives are indexed
triangles with the same attributes and a material. Files with extensions, skins, animations, morph targets or external
images are rejected with an error. MP3D `.glb` and HM3D `.basis.glb` files meet these conditions.

The output uses 32-bit indices, so files grow by about 2% (MP3D) to 6% (HM3D).

General-purpose tools such as [gltfpack](https://github.com/zeux/meshoptimizer/tree/master/gltf) also merge primitives
by material, but they also re-index vertices and reorder triangles, so their output is not geometry-identical to the
input.

## Data license

The code is released under the MIT license. Matterport3D and [HM3D](https://aihabitat.org/datasets/hm3d/) are provided
by Matterport under their own terms; see the [MP3D terms of use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).
This repository contains no scene data. Merged files are derived from the datasets and fall under the same terms as the
originals.
