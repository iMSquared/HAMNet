#!/usr/bin/env python3

import numpy as np
import json

from ham.env.scene.object_set import ObjectSet
from ham.env.scene.dgn_object_set import DGNObjectSet
from ham.env.scene.mesh_object_set import MeshObjectSet
from ham.env.scene.filter_object_set import FilteredObjectSet, FilterDims
from ham.env.scene.combine_object_set import CombinedObjectSet


def is_thin(extent, threshold: float = 2.5):
    size = np.sort(extent)
    return (size[1] >= threshold * size[0])


def load_obj_set(self) -> ObjectSet:
    cfg = self.cfg
    allowlist = None

    assert (len(cfg.base_set) > 0)

    obj_sets = []
    for base_set in cfg.base_set:
        if base_set == 'dgn':
            meta = DGNObjectSet(cfg.dgn)
        elif base_set == 'mesh':
            meta = MeshObjectSet(cfg.mesh)
        else:
            raise KeyError(F'Unknown base object set = {cfg.base_set}')
        keys = meta.keys()
        print(F'init : {len(keys)}')

        # First, filter by availability of all required fields.
        # Determine required attributes...
        need_attr = list(cfg.need_attr)
        if need_attr is None:
            need_attr = []
        if cfg.goal_type == 'stable':
            need_attr.append('pose')
        for attr in need_attr:
            query = getattr(meta, attr)
            fkeys = []
            for key in keys:
                try:
                    if query(key) is None:
                        continue
                except KeyError:
                    continue
                fkeys.append(key)
            keys = fkeys
        print(F'after filter by "need" : {len(keys)}')

        # Filter by `filter_class`.
        if cfg.filter_class is not None:
            keys = [key for key in keys
                    if meta.label(key) in cfg.filter_class]
        print(F'after filter by "class" : {len(keys)}')

        if cfg.filter_key is not None:
            keys = [key for key in keys
                    if key in cfg.filter_key]
        print(F'after filter by "filter_key" : {len(keys)}')

        if cfg.filter_file is not None:
            with open(cfg.filter_file, 'r') as fp:
                allowlist = [str(s) for s in json.load(fp)]
            keys = [key for key in keys
                    if key in allowlist]
        print(F'after filter by "filter_file" : {len(keys)}')

        # Filter by size, and remove degenerate mesh
        if cfg.filter_complex:
            keys = [key for key in keys if
                    (meta.num_verts(key) < cfg.max_vertex_count and
                        meta.num_hulls(key) < cfg.max_chull_count and
                        key != '4Shelves_fd0fd7b2c19cce39e3783ec57dd5d298_0.001818238917007018')
                    ]
        print(F'after filter by "complex" : {len(keys)}')

        if cfg.filter_dims is not None:
            d_min, d_max, r_max = cfg.filter_dims
            f = FilterDims(d_min, d_max, r_max)
            keys = [key for key in keys if f(meta, key)]
        print(F'after filter by "dims" : {len(keys)}')

        # Filter by size, and remove degenerate mesh
        if cfg.filter_pose_count:
            pmin, pmax = cfg.filter_pose_count
            keys = [key for key in keys if
                    meta.pose(key) is not None
                    and pmin <= meta.pose(key).shape[0]
                    and meta.pose(key).shape[0] < pmax
                    ]
        print(F'after filter by "pose_count" : {len(keys)}')
        if cfg.filter_thin:
            keys = [key for key in keys
                    if not is_thin(meta.obb(key)[1],
                                   threshold=cfg.thin_threshold)
                    ]
            print(F'after filter by "thin logic" : {len(keys)}')

        obj_sets.append(FilteredObjectSet(meta, keys=keys))
    if len(obj_sets) == 1:
        return obj_sets[0]
    return CombinedObjectSet(obj_sets)
