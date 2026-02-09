#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np


def _write_bin_header(f, nrows: int, dim: int) -> None:
    f.write(struct.pack('<II', int(nrows), int(dim)))


def _write_fbin_from_hdf5_dataset(ds: h5py.Dataset, out_path: Path, chunk_rows: int = 4096) -> None:
    if ds.ndim != 2:
        raise ValueError(f"expected 2D dataset, got shape={ds.shape}")

    nrows, dim = ds.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'wb') as f:
        _write_bin_header(f, nrows, dim)

        for start in range(0, nrows, chunk_rows):
            end = min(nrows, start + chunk_rows)
            block = ds[start:end, :]
            block = np.asarray(block, dtype=np.float32, order='C')
            f.write(block.tobytes(order='C'))


def _write_ibin_from_hdf5_dataset(ds: h5py.Dataset, out_path: Path, chunk_rows: int = 4096) -> None:
    if ds.ndim != 2:
        raise ValueError(f"expected 2D dataset, got shape={ds.shape}")

    nrows, dim = ds.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'wb') as f:
        _write_bin_header(f, nrows, dim)

        for start in range(0, nrows, chunk_rows):
            end = min(nrows, start + chunk_rows)
            block = ds[start:end, :]
            block = np.asarray(block, dtype=np.uint32, order='C')
            f.write(block.tobytes(order='C'))


def main() -> int:
    ap = argparse.ArgumentParser(description='Convert ann-benchmarks-style HDF5 to DiskANN .bin format')
    ap.add_argument('--hdf5', required=True, help='Path to an ann-benchmarks-style .hdf5 file')
    ap.add_argument('--out-dir', required=True, help='Output directory for train.fbin/test.fbin/neighbors.ibin')
    ap.add_argument('--train-key', default='train', help='HDF5 key for base vectors (default: train)')
    ap.add_argument('--test-key', default='test', help='HDF5 key for queries (default: test)')
    ap.add_argument('--neighbors-key', default='neighbors', help='HDF5 key for ground-truth neighbors (default: neighbors)')
    ap.add_argument('--chunk-rows', type=int, default=4096, help='Rows per chunk when streaming from HDF5')
    args = ap.parse_args()

    hdf5_path = Path(args.hdf5)
    out_dir = Path(args.out_dir)
    chunk_rows = int(args.chunk_rows)

    if not hdf5_path.is_file():
        raise FileNotFoundError(str(hdf5_path))

    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        required = [args.train_key, args.test_key, args.neighbors_key]
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"missing keys in hdf5: {missing}; found={list(f.keys())}")

        train = f[args.train_key]
        test = f[args.test_key]
        neighbors = f[args.neighbors_key]

        if train.ndim != 2 or test.ndim != 2:
            raise ValueError(f"train/test must be 2D; got train.shape={train.shape}, test.shape={test.shape}")
        if int(train.shape[1]) != int(test.shape[1]):
            raise ValueError(f"train/test dim mismatch: train dim={train.shape[1]}, test dim={test.shape[1]}")
        if neighbors.ndim != 2:
            raise ValueError(f"neighbors must be 2D; got neighbors.shape={neighbors.shape}")
        if int(neighbors.shape[0]) != int(test.shape[0]):
            raise ValueError(
                f"neighbors rows must match test rows: neighbors rows={neighbors.shape[0]}, test rows={test.shape[0]}"
            )

        # Basic sanity
        if train.dtype != np.float32 or test.dtype != np.float32:
            # We still cast, but warn via exception message if unexpected.
            pass
        if neighbors.dtype.kind not in ('i', 'u'):
            raise ValueError(f"neighbors dtype must be integer, got {neighbors.dtype}")

        print('Writing train.fbin ...')
        _write_fbin_from_hdf5_dataset(train, out_dir / 'train.fbin', chunk_rows=chunk_rows)

        print('Writing test.fbin ...')
        _write_fbin_from_hdf5_dataset(test, out_dir / 'test.fbin', chunk_rows=chunk_rows)

        print('Writing neighbors.ibin ...')
        _write_ibin_from_hdf5_dataset(neighbors, out_dir / 'neighbors.ibin', chunk_rows=chunk_rows)

        print('Done.')
        print('Output dir:', str(out_dir))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
