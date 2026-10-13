"""Recompute the navmesh from the original and from the merged scene mesh and compare.

The .navmesh files shipped with the datasets are not touched by merge_glb.py, so this only
matters when habitat rebuilds the navmesh from the render mesh. habitat-sim 0.1.7 does that
automatically whenever the agent height/radius differ from the defaults (1.5 m / 0.1 m),
e.g. the VLN-CE RxR setup (0.88 m / 0.18 m).

Tested with habitat-sim 0.1.7 and 0.3.0.

Usage:
    python tools/check_navmesh.py original.glb merged.glb [--height 0.88 --radius 0.18]
    python tools/check_navmesh.py --scenes-dir data/scene_datasets/mp3d
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile

import habitat_sim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_render import linked_pair  # noqa: E402


def build(scene, height, radius, out_path, dataset_config):
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = scene
    sc.create_renderer = False
    if dataset_config:
        sc.scene_dataset_config_file = dataset_config
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [habitat_sim.agent.AgentConfiguration()]))
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height, settings.agent_radius = height, radius
    try:
        sim.recompute_navmesh(sim.pathfinder, settings)                                   # >= 0.2
    except TypeError:
        sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=False)     # 0.1.x
    sim.pathfinder.save_nav_mesh(out_path)
    area = sim.pathfinder.navigable_area
    sim.close()
    with open(out_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest(), area


def check_scene(orig_path, merged_path, height, radius, dataset_config):
    with tempfile.TemporaryDirectory() as tmp:
        lo, lm = linked_pair(orig_path, merged_path, tmp)
        o = build(lo, height, radius, os.path.join(tmp, 'o.navmesh'), dataset_config)
        m = build(lm, height, radius, os.path.join(tmp, 'm.navmesh'), dataset_config)
    return {'scene': os.path.basename(merged_path), 'identical': o[0] == m[0],
            'area_orig': round(o[1], 4), 'area_merged': round(m[1], 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('orig', nargs='?')
    ap.add_argument('merged', nargs='?')
    ap.add_argument('--scenes-dir')
    ap.add_argument('--pattern', default='*/*.glb')
    ap.add_argument('--dataset-config')
    ap.add_argument('--height', type=float, default=0.88)
    ap.add_argument('--radius', type=float, default=0.18)
    args = ap.parse_args()
    if args.scenes_dir:
        pairs = [(p + '.orig', p) for p in sorted(glob.glob(os.path.join(args.scenes_dir, args.pattern)))
                 if os.path.exists(p + '.orig')]
    elif args.orig and args.merged:
        pairs = [(args.orig, args.merged)]
    else:
        ap.error('give orig and merged, or --scenes-dir')
    results = [check_scene(o, m, args.height, args.radius, args.dataset_config) for o, m in pairs]
    for r in results:
        print(json.dumps(r), flush=True)
    print('%d scenes, identical navmesh %d' % (len(results), sum(r['identical'] for r in results)), file=sys.stderr)


if __name__ == '__main__':
    main()
