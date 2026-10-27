"""Synchronous reader for this competition's numeric GEFF v3 arrays.

Avoids the Zarr background event loop (which stalls in the restricted runner).
The division scoring algorithm itself is still the official implementation.
Unsupported codecs/layouts fail explicitly instead of silently decoding them.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from numcodecs import Zstd


def read_array(path):
    path = Path(path)
    meta = json.loads((path / 'zarr.json').read_text())
    codecs = meta['codecs']
    if (meta['zarr_format'] != 3 or meta['node_type'] != 'array'
            or meta['chunk_grid']['name'] != 'regular'
            or meta['chunk_key_encoding']['name'] != 'default'
            or [c['name'] for c in codecs] != ['bytes', 'zstd']
            or meta.get('storage_transformers', [])):
        raise ValueError(f'Unsupported competition array layout: {path}')
    dtype = np.dtype(meta['data_type']).newbyteorder(
        '<' if codecs[0].get('configuration', {}).get('endian', 'little') == 'little' else '>')
    shape = tuple(meta['shape'])
    chunk = tuple(meta['chunk_grid']['configuration']['chunk_shape'])
    result = np.full(shape, meta['fill_value'], dtype=dtype)
    separator = meta['chunk_key_encoding'].get('configuration', {}).get('separator', '/')
    for index in itertools.product(*(range((s+c-1)//c) for s, c in zip(shape, chunk))):
        key = separator.join(['c', *map(str, index)])
        file = path / key
        if not file.exists():
            continue  # absent Zarr chunks have the declared fill value
        data = np.frombuffer(Zstd().decode(file.read_bytes()), dtype=dtype).reshape(chunk)
        target = tuple(slice(i*c, min((i+1)*c, s)) for i, c, s in zip(index, chunk, shape))
        source = tuple(slice(0, sl.stop-sl.start) for sl in target)
        result[target] = data[source]
    return result


def load_gt_sync(path):
    path = Path(path)
    ids = read_array(path / 'nodes/ids')
    t = read_array(path / 'nodes/props/t/values')
    xyz = np.column_stack([read_array(path / f'nodes/props/{key}/values') for key in ('z', 'y', 'x')])
    edges = read_array(path / 'edges/ids') if (path / 'edges/ids').exists() else np.empty((0, 2), int)
    attrs = json.loads((path / 'zarr.json').read_text())['attributes']
    total = attrs.get('geff', {}).get('extra', {}).get('estimated_number_of_nodes')
    return ids, t, xyz, edges, float(total) if total else None
