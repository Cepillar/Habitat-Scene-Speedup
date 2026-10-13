"""Check that a merged .glb holds exactly the same geometry as the original.

For every triangle the raw bytes of all vertex attributes of its three vertices are
compared, per material. Two verdicts are reported:

  same_geometry   every material has the same triangles, in the same order within the
                  material; texture bytes, materials, samplers and the world transform
                  are unchanged.
  same_order      additionally, the global triangle order (the order in which habitat
                  draws them and feeds them to the navmesh builder) is unchanged. This
                  holds when the original file already lists its primitives grouped by
                  material. Rendering and navmesh recomputation are then identical by
                  construction. When it does not hold, coplanar overlapping faces with
                  different materials can resolve differently, so a few pixels may differ.

No GPU or habitat-sim needed.

Usage:
    python tools/check_geometry.py original.glb merged.glb
    python tools/check_geometry.py --scenes-dir data/scene_datasets/mp3d   # <name>.orig vs <name>
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from merge_glb import read_glb, acc_array, node_world_transforms  # noqa: E402


def mesh_prims(js):
    """(world matrix, primitive) for every primitive, in scene-graph traversal order."""
    world = node_world_transforms(js)
    out = []
    def walk(ni):
        nd = js['nodes'][ni]
        if 'mesh' in nd:
            for pr in js['meshes'][nd['mesh']]['primitives']:
                out.append((world[ni], pr))
        for c in nd.get('children', []):
            walk(c)
    for r in js['scenes'][js.get('scene', 0)]['nodes']:
        walk(r)
    return out


def triangle_rows(js, bin_, prim, keys):
    """One row per triangle: the attribute bytes of its three vertices, winding kept."""
    idx = acc_array(js, bin_, prim['indices']).reshape(-1).astype(np.int64)
    verts = np.concatenate([acc_array(js, bin_, prim['attributes'][k]).view(np.uint8) for k in keys], 1)
    return verts[idx].reshape(len(idx) // 3, -1)


def image_hashes(js, bin_):
    out = []
    for im in js.get('images', []):
        bv = js['bufferViews'][im['bufferView']]
        s = bv.get('byteOffset', 0)
        out.append((im.get('mimeType'), hashlib.md5(bin_[s:s + bv['byteLength']]).hexdigest()))
    return out


def load(path):
    js, bin_ = read_glb(path)
    prims = mesh_prims(js)
    keys = sorted(prims[0][1]['attributes'])
    rows = [(pr['material'], triangle_rows(js, bin_, pr, keys)) for _, pr in prims]
    return js, bin_, prims, keys, rows


def compare(orig_path, merged_path):
    jo, bo, po, ko, ro = load(orig_path)
    jm, bm, pm, km, rm = load(merged_path)
    per_mat = lambda rows: {m: np.concatenate([r for mm, r in rows if mm == m]) for m in sorted({m for m, _ in rows})}
    go, gm = per_mat(ro), per_mat(rm)
    same_geometry = (ko == km and sorted(go) == sorted(gm)
                     and all(np.array_equal(go[m], gm[m]) for m in go)
                     and all(np.allclose(w, pm[0][0], atol=1e-9) for w, _ in po + pm)
                     and all(jo.get(k) == jm.get(k) for k in ('materials', 'textures', 'samplers'))
                     and image_hashes(jo, bo) == image_hashes(jm, bm))
    same_order = same_geometry and np.array_equal(np.concatenate([r for _, r in ro]), np.concatenate([r for _, r in rm]))
    return {'scene': os.path.basename(merged_path), 'primitives_before': len(po), 'primitives_after': len(pm),
            'triangles': int(sum(len(r) for _, r in ro)), 'same_geometry': bool(same_geometry),
            'same_order': bool(same_order)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('orig', nargs='?')
    ap.add_argument('merged', nargs='?')
    ap.add_argument('--scenes-dir')
    ap.add_argument('--pattern', default='*/*.glb')
    args = ap.parse_args()
    if args.scenes_dir:
        pairs = [(p + '.orig', p) for p in sorted(glob.glob(os.path.join(args.scenes_dir, args.pattern)))
                 if os.path.exists(p + '.orig')]
    elif args.orig and args.merged:
        pairs = [(args.orig, args.merged)]
    else:
        ap.error('give orig and merged, or --scenes-dir')
    results = []
    for o, m in pairs:
        r = compare(o, m)
        results.append(r)
        print(json.dumps(r), flush=True)
    bad = [r['scene'] for r in results if not r['same_geometry']]
    print('%d scenes: same geometry %d, same global order %d%s' % (
        len(results), len(results) - len(bad), sum(r['same_order'] for r in results),
        ('; DIFFERENT: ' + ' '.join(bad)) if bad else ''), file=sys.stderr)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
