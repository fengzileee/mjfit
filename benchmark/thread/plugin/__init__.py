"""Build and load the analytic ISO thread SDF plugin (iso_thread_sdf.cc)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco

SRC = Path(__file__).resolve().parent / 'iso_thread_sdf.cc'
NAME = 'iso.sdf.thread'
_loaded = False


def library() -> Path:
    """Compile against the installed mujoco wheel (once per machine and
    version); the library lives in the gitignored cache."""
    from ...representations import CACHE_DIR
    pkg = Path(mujoco.__file__).resolve().parent
    out = CACHE_DIR / f'libiso_thread_sdf_{mujoco.__version__}.so'
    if not out.exists() or out.stat().st_mtime < SRC.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        lib = next(pkg.glob('libmujoco.so*'))
        subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', f'-I{pkg / "include"}',
                        str(SRC), str(lib), f'-Wl,-rpath,{pkg}', '-o', str(out)], check=True)
    return out


def load() -> str:
    """Register the plugin with MuJoCo (idempotent); returns its name."""
    global _loaded
    if not _loaded:
        mujoco.mj_loadPluginLibrary(str(library()))
        _loaded = True
    return NAME
