"""Merge all glTF primitives that share a material into a single primitive.

Habitat-sim draws every primitive as a separate draw call, and rendering of
the Matterport3D / HM3D scene meshes is limited by the CPU cost of issuing
those calls. The original MP3D files split the house into small primitives
(uint16 indices, at most 65535 vertices each), so some scenes have thousands
of them. This script concatenates the vertex and index buffers of each
material into one primitive. Vertex values, triangles, materials and texture
bytes are copied unchanged.

Usage:
    python merge_glb.py in.glb out.glb
    python merge_glb.py --scenes-dir data/scene_datasets/mp3d          # in place, keeps <name>.orig
    python merge_glb.py --scenes-dir data/scene_datasets/mp3d --restore
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

GLB_MAGIC, CHUNK_JSON, CHUNK_BIN = 0x46546C67, 0x4E4F534A, 0x004E4942
CT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_N = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def read_glb(path):
    with open(path, 'rb') as f:
        magic, ver, total = struct.unpack('<III', f.read(12))
        assert magic == GLB_MAGIC and ver == 2, 'not a glTF 2.0 binary file'
        chunks = {}
        while f.tell() < total:
            clen, ctype = struct.unpack('<II', f.read(8))
            chunks[ctype] = f.read(clen)
    return json.loads(chunks[CHUNK_JSON]), chunks.get(CHUNK_BIN, b'')


def acc_array(js, bin_, ai):
    a = js['accessors'][ai]
    assert 'sparse' not in a, 'sparse accessors are not supported'
    assert not a.get('normalized'), 'normalized accessors are not supported'
    bv = js['bufferViews'][a['bufferView']]
    n, ncomp = a['count'], TYPE_N[a['type']]
    dt = np.dtype(CT_DTYPE[a['componentType']])
    tight = ncomp * dt.itemsize
    base = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = bv.get('byteStride', tight) or tight
    raw = np.frombuffer(bin_, np.uint8)
    if stride == tight:
        arr = raw[base:base + n * tight].view(dt).reshape(n, ncomp)
    else:
        idx = base + np.arange(n)[:, None] * stride + np.arange(tight)
        arr = raw[idx].reshape(-1).view(dt).reshape(n, ncomp)
    return np.ascontiguousarray(arr)


def node_world_transforms(js):
    """Return {node id: 4x4 world matrix} for every node that carries a mesh."""
    def local_mat(nd):
        if 'matrix' in nd:
            return np.array(nd['matrix'], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        if 'scale' in nd:
            m = m @ np.diag(list(nd['scale']) + [1.0])
        if 'rotation' in nd:
            x, y, z, w = nd['rotation']
            r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            rm = np.eye(4); rm[:3, :3] = r; m = rm @ m
        if 'translation' in nd:
            tm = np.eye(4); tm[:3, 3] = nd['translation']; m = tm @ m
        return m
    out = {}
    scene = js['scenes'][js.get('scene', 0)]
    def walk(ni, parent):
        nd = js['nodes'][ni]
        w = parent @ local_mat(nd)
        if 'mesh' in nd:
            out[ni] = w
        for c in nd.get('children', []):
            walk(c, w)
    for root in scene['nodes']:
        walk(root, np.eye(4))
    return out


def merge(src, dst, verbose=True):
    js, bin_ = read_glb(src)
    # Anything this script would otherwise drop or mis-handle is rejected up front.
    assert not js.get('animations') and not js.get('skins'), 'animations/skins are not supported'
    assert not js.get('extensionsUsed') and not js.get('extensionsRequired'), \
        'glTF extensions are not supported: %s' % js.get('extensionsUsed')
    assert all('bufferView' in im for im in js.get('images', [])), 'external image URIs are not supported'

    # All mesh nodes must share one world transform; it is kept on the single output node.
    wt = node_world_transforms(js)
    mats44 = list(wt.values())
    for m in mats44[1:]:
        assert np.allclose(m, mats44[0], atol=1e-9), 'mesh nodes have different transforms'
    common = mats44[0]

    prims = []
    for ni in wt:
        for p in js['meshes'][js['nodes'][ni]['mesh']]['primitives']:
            assert p.get('mode', 4) == 4, 'only TRIANGLES primitives are supported'
            assert 'targets' not in p, 'morph targets are not supported'
            prims.append(p)
    attr_keys = sorted(prims[0]['attributes'].keys())
    for p in prims:
        assert sorted(p['attributes'].keys()) == attr_keys, 'primitives have different attribute sets'
        assert 'indices' in p and 'material' in p, 'every primitive needs indices and a material'

    groups = {}
    for p in prims:
        groups.setdefault(p['material'], []).append(p)

    merged = {}  # material -> {attribute: array, 'indices': array}
    tri_before = 0
    for mat, ps in sorted(groups.items()):
        arrs = {k: [] for k in attr_keys}
        idxs = []
        vbase = 0
        for p in ps:
            counts = set()
            for k in attr_keys:
                a = acc_array(js, bin_, p['attributes'][k])
                arrs[k].append(a)
                counts.add(len(a))
            assert len(counts) == 1, 'attributes of one primitive have different vertex counts'
            ind = acc_array(js, bin_, p['indices']).reshape(-1).astype(np.uint32)
            idxs.append(ind + vbase)
            vbase += counts.pop()
            tri_before += len(ind) // 3
        for k in attr_keys:
            assert len(set(a.dtype for a in arrs[k])) == 1, 'attribute %s has mixed component types' % k
        merged[mat] = {k: np.concatenate(arrs[k]) for k in attr_keys}
        merged[mat]['indices'] = np.concatenate(idxs)

    # New binary chunk: images first (bytes unchanged), then geometry.
    new_bvs, new_bin = [], []
    off = 0
    def push(data, target=None, align=4):
        nonlocal off
        pad = (-off) % align
        if pad:
            new_bin.append(b'\x00' * pad); off += pad
        bv = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target:
            bv['target'] = target
        new_bvs.append(bv); new_bin.append(data); off += len(data)
        return len(new_bvs) - 1

    img_remap = {}
    for i, img in enumerate(js.get('images', [])):
        bv = js['bufferViews'][img['bufferView']]
        s = bv.get('byteOffset', 0)
        img_remap[i] = push(bin_[s:s + bv['byteLength']])

    new_accs, new_prims = [], []
    def add_acc(arr, ctype, typ, target):
        bvi = push(np.ascontiguousarray(arr).tobytes(), target=target)
        acc = {'bufferView': bvi, 'byteOffset': 0, 'componentType': ctype,
               'count': int(arr.shape[0]), 'type': typ}
        if typ != 'SCALAR':
            acc['min'] = [float(x) for x in arr.min(0)]
            acc['max'] = [float(x) for x in arr.max(0)]
        new_accs.append(acc)
        return len(new_accs) - 1

    tri_after = 0
    for mat in sorted(merged):
        g = merged[mat]
        attrs = {}
        for k in attr_keys:
            a = g[k]
            ct = {np.dtype(np.float32): 5126, np.dtype(np.uint16): 5123, np.dtype(np.uint32): 5125,
                  np.dtype(np.uint8): 5121}[a.dtype]
            attrs[k] = add_acc(a, ct, {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4'}[a.shape[1]], 34962)
        ii = add_acc(g['indices'].reshape(-1, 1), 5125, 'SCALAR', 34963)
        new_accs[ii].pop('min', None); new_accs[ii].pop('max', None)
        new_prims.append({'attributes': attrs, 'indices': ii, 'material': mat, 'mode': 4})
        tri_after += len(g['indices']) // 3
    assert tri_before == tri_after, 'triangle count changed'

    out = {'asset': {'version': '2.0', 'generator': 'Habitat-Scene-Speedup merge_glb.py'},
           'scene': 0, 'scenes': [{'nodes': [0]}]}
    node = {'mesh': 0, 'name': 'merged'}
    if not np.allclose(common, np.eye(4)):
        node['matrix'] = [float(x) for x in common.T.reshape(-1)]
    out['nodes'] = [node]
    out['meshes'] = [{'primitives': new_prims}]
    for k in ('materials', 'textures', 'samplers'):
        if k in js:
            out[k] = js[k]
    if 'images' in js:
        out['images'] = []
        for i, img in enumerate(js['images']):
            im = dict(img); im['bufferView'] = img_remap[i]
            out['images'].append(im)
    out['accessors'] = new_accs
    out['bufferViews'] = new_bvs
    bin_all = b''.join(new_bin)
    out['buffers'] = [{'byteLength': len(bin_all)}]

    jb = json.dumps(out, separators=(',', ':')).encode()
    jb += b' ' * ((-len(jb)) % 4)
    bb = bin_all + b'\x00' * ((-len(bin_all)) % 4)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    with open(dst, 'wb') as f:
        f.write(struct.pack('<III', GLB_MAGIC, 2, total))
        f.write(struct.pack('<II', len(jb), CHUNK_JSON)); f.write(jb)
        f.write(struct.pack('<II', len(bb), CHUNK_BIN)); f.write(bb)
    if verbose:
        name = os.path.basename(src)
        name = name[:-len('.orig')] if name.endswith('.orig') else name
        print('%s: %d -> %d primitives, %d triangles, %.1f MB' % (name, len(prims), len(new_prims), tri_after, total / 1e6))
    return len(prims), len(new_prims)


def scene_files(root, pattern):
    # A scene counts if either <name> or <name>.orig exists (a run may have been interrupted
    # between the rename and the write).
    files = set(glob.glob(os.path.join(root, pattern)))
    files |= {f[:-len('.orig')] for f in glob.glob(os.path.join(root, pattern + '.orig'))}
    return sorted(f for f in files if not f.endswith('.orig'))


def merge_dir(root, pattern):
    total_before = total_after = 0
    for path in scene_files(root, pattern):
        orig = path + '.orig'
        if not os.path.exists(orig):
            os.rename(path, orig)
        tmp = path + '.tmp'
        before, after = merge(orig, tmp)
        os.replace(tmp, path)
        total_before += before; total_after += after
    print('total: %d -> %d primitives' % (total_before, total_after))


def restore_dir(root, pattern):
    for path in scene_files(root, pattern):
        orig = path + '.orig'
        if os.path.exists(orig):
            os.replace(orig, path)
            print('restored', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='?')
    ap.add_argument('dst', nargs='?')
    ap.add_argument('--scenes-dir', help='process every scene under this directory in place')
    ap.add_argument('--pattern', default='*/*.glb', help='glob under --scenes-dir (default: */*.glb)')
    ap.add_argument('--restore', action='store_true', help='move <name>.orig back over <name>')
    args = ap.parse_args()
    if args.scenes_dir:
        (restore_dir if args.restore else merge_dir)(args.scenes_dir, args.pattern)
    elif args.src and args.dst:
        merge(args.src, args.dst)
    else:
        ap.error('give src and dst, or --scenes-dir')


if __name__ == '__main__':
    sys.exit(main())
