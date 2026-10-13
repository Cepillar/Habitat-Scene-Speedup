"""Render the original and the merged scene from the same poses, compare pixels and time it.

For each scene, N random navigable poses are drawn (fixed seed) and rendered with both
files through six cameras: RGB 224x224 and depth 256x256 at 1.25 m (the VLN-CE R2R setup),
RGB and depth 640x480 at 0.88 m (RxR), and RGB 224x224 pitched +-30 degrees. Every pixel is
compared. Each camera is timed separately as draw + read back to the CPU, so the GPU work
is included; the first pass over the poses is a warm-up and not timed.

Tested with habitat-sim 0.1.7 and 0.3.0.

Usage:
    python tools/check_render.py original.glb merged.glb
    python tools/check_render.py --scenes-dir data/scene_datasets/mp3d       # <name>.orig vs <name>
    python tools/check_render.py --scenes-dir data/scene_datasets/hm3d/val --pattern '*/*.basis.glb'
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

# (name, color or depth, width, height, hfov, camera height, pitch in degrees)
CAMERAS = [
    ('rgb_224', 'color', 224, 224, 90, 1.25, 0), ('depth_256', 'depth', 256, 256, 90, 1.25, 0),
    ('rgb_640x480', 'color', 640, 480, 79, 0.88, 0), ('depth_640x480', 'depth', 640, 480, 79, 0.88, 0),
    ('rgb_224_up', 'color', 224, 224, 90, 1.25, 30), ('rgb_224_down', 'color', 224, 224, 90, 1.25, -30),
]


def sensor_spec(name, kind, w, h, hfov, y, pitch):
    typ = habitat_sim.SensorType.COLOR if kind == 'color' else habitat_sim.SensorType.DEPTH
    if hasattr(habitat_sim, 'CameraSensorSpec'):      # habitat-sim >= 0.2
        s = habitat_sim.CameraSensorSpec()
        s.hfov = float(hfov)
    else:                                             # habitat-sim 0.1.x
        s = habitat_sim.SensorSpec()
        s.parameters['hfov'] = str(hfov)
    s.uuid, s.sensor_type, s.resolution = name, typ, [h, w]
    s.position = [0.0, y, 0.0]
    s.orientation = [np.deg2rad(pitch), 0.0, 0.0]
    return s


def linked_pair(orig_path, merged_path, tmpdir):
    """Link both files into their own temp dirs under the merged file's name, each next to the
    same navmesh. habitat picks the loader from the extension (so '<name>.orig' must be renamed)
    and matches stages in a dataset config by path, so both versions are loaded the same way."""
    nav = os.path.splitext(merged_path)[0] + '.navmesh'
    out = []
    for tag, path in (('orig', orig_path), ('merged', merged_path)):
        d = os.path.join(tmpdir, tag)
        os.makedirs(d)
        link = os.path.join(d, os.path.basename(merged_path))
        os.symlink(os.path.abspath(path), link)
        if os.path.exists(nav):
            os.symlink(os.path.abspath(nav), os.path.splitext(link)[0] + '.navmesh')
        out.append(link)
    return out


def make_sim(scene, gpu, dataset_config):
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = scene
    sc.gpu_device_id = gpu
    if dataset_config:
        sc.scene_dataset_config_file = dataset_config
    ac = habitat_sim.agent.AgentConfiguration()   # default size: the .navmesh file is loaded as is
    ac.sensor_specifications = [sensor_spec(*c) for c in CAMERAS]
    return habitat_sim.Simulator(habitat_sim.Configuration(sc, [ac]))


def render(sim, poses, rounds=3):
    assert rounds >= 2, 'the first round is a warm-up and is not timed'
    agent = sim.get_agent(0)
    frames, ms = [], {c[0]: 0.0 for c in CAMERAS}
    for r in range(rounds):
        for pos, yaw in poses:
            st = habitat_sim.AgentState()
            st.position = pos
            st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            agent.set_state(st)
            obs = {}
            for c in CAMERAS:
                sensor = sim._sensors[c[0]]
                t0 = time.perf_counter()
                sensor.draw_observation()
                o = sensor.get_observation()
                if r > 0:
                    ms[c[0]] += time.perf_counter() - t0
                obs[c[0]] = np.array(o)
            if r == 0:
                frames.append(obs)
    n = len(poses) * (rounds - 1)
    return frames, {k: 1000 * v / n for k, v in ms.items()}


def check_scene(orig_path, merged_path, gpu, n_poses, dataset_config):
    res, frames, poses = {'scene': os.path.basename(merged_path)}, {}, None
    with tempfile.TemporaryDirectory() as tmp:
        files = dict(zip(('orig', 'merged'), linked_pair(orig_path, merged_path, tmp)))
        for ver in ('orig', 'merged'):
            t0 = time.perf_counter()
            sim = make_sim(files[ver], gpu, dataset_config)
            res['load_s_' + ver] = round(time.perf_counter() - t0, 3)
            if poses is None:
                sim.pathfinder.seed(0)
                rng = np.random.RandomState(0)
                poses = [(np.array(sim.pathfinder.get_random_navigable_point()), rng.uniform(0, 2 * np.pi))
                         for _ in range(n_poses)]
            frames[ver], res['ms_' + ver] = render(sim, poses)
            sim.close()
    res['pixels_differing'] = {}
    for c in CAMERAS:
        a = np.stack([f[c[0]] for f in frames['orig']])
        b = np.stack([f[c[0]] for f in frames['merged']])
        neq = a != b
        res['pixels_differing'][c[0]] = int((neq.any(-1) if neq.ndim == 4 else neq).sum())
    res['identical'] = not any(res['pixels_differing'].values())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('orig', nargs='?')
    ap.add_argument('merged', nargs='?')
    ap.add_argument('--scenes-dir')
    ap.add_argument('--pattern', default='*/*.glb')
    ap.add_argument('--dataset-config', help='scene_dataset_config.json, if you load scenes through one')
    ap.add_argument('--poses', type=int, default=50)
    ap.add_argument('--gpu', type=int, default=0)
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
        r = check_scene(o, m, args.gpu, args.poses, args.dataset_config)
        results.append(r)
        print(json.dumps(r), flush=True)
    view = lambda ms: ms['rgb_224'] + ms['depth_256']
    before = np.mean([view(r['ms_orig']) for r in results])
    after = np.mean([view(r['ms_merged']) for r in results])
    print('%d scenes, pixel-identical %d; RGB 224 + depth 256 per view: %.3f -> %.3f ms (mean over scenes)' % (
        len(results), sum(r['identical'] for r in results), before, after), file=sys.stderr)


if __name__ == '__main__':
    main()
