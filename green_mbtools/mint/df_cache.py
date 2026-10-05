"""Provenance checks for GREEN's internal periodic integral caches."""
import hashlib
import json
import logging

import h5py
import numpy as np
import pyscf
from pyscf.df import addons


SCHEMA_VERSION = 1
FRAME_VERSION = 1
GROUP = "green_cache"


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def cache_request(mydf, role="ordinary"):
    """Record physical/build inputs, excluding paths, memory and logging options."""
    if role not in {"ordinary", "ewald"}:
        raise ValueError("Unsupported DF cache interaction role: " + role)
    cell = mydf.cell
    auxcell = addons.make_auxmol(cell, mydf.auxbasis)
    backend = getattr(mydf, "_green_df_backend",
                      "ccgdf" if mydf._prefer_ccdf else "rsgdf")
    return _plain({
        "backend": backend, "role": role, "pyscf_version": pyscf.__version__,
        "frame_version": FRAME_VERSION if backend == "rsgdf" and role == "ordinary" else 0,
        "cell": {
            "lattice": cell.lattice_vectors(), "atoms": cell._atom,
            "basis": cell._basis, "pseudo": cell._pseudo, "ecp": cell._ecp,
            "charge": cell.charge, "spin": cell.spin, "cart": cell.cart,
            "dimension": cell.dimension, "low_dim_ft_type": cell.low_dim_ft_type,
            "omega": cell.omega, "precision": cell.precision,
            "mesh": cell.mesh, "rcut": cell.rcut,
        },
        "auxiliary": {"basis": auxcell._basis, "naux": auxcell.nao_nr(),
                      "exp_to_discard": mydf.exp_to_discard},
        "kpts": mydf.kpts, "kpts_band": mydf.kpts_band,
        "settings": {
            "df_mesh": mydf.mesh, "eta": mydf.eta,
            "linear_dep_threshold": mydf.linear_dep_threshold,
            "eigenvalue": bool(getattr(mydf, "use_j2c_eig_decomposition", True)),
            "j_only": bool(mydf._j_only),
            "space_symm": bool(getattr(mydf, "space_symm", False)),
            "tr_symm": bool(getattr(mydf, "tr_symm", False)),
            "x2c": int(getattr(mydf, "x2c", 0)),
        },
    })


def write_cache_provenance(mydf, path, role, producer, effective_mesh=None):
    """Called only after a successful new build; never relabel a reused file."""
    request = _json(cache_request(mydf, role))
    with h5py.File(path, "a") as f:
        group = f.require_group(GROUP)
        group.attrs["schema_version"] = SCHEMA_VERSION
        group.attrs["producer"] = producer
        group.attrs["request_sha256"] = hashlib.sha256(request.encode()).hexdigest()
        group.attrs["effective_mesh"] = _json(effective_mesh)
        if "request" in group:
            del group["request"]
        group.create_dataset("request", data=request, dtype=h5py.string_dtype("utf-8"))


def _mismatch(path, detail):
    raise RuntimeError(
        f"DF cache mismatch for {path}: {detail}. Start a fresh calculation directory; "
        "the original cache and exports have been preserved."
    )


def validate_df_cache(mydf, path, role="ordinary"):
    """Validate a managed cache read-only before reuse or replacing export files."""
    expected = cache_request(mydf, role)
    with h5py.File(path, "r") as f:
        if GROUP not in f:
            # Old CCGDF files remain available through an explicit compatibility
            # option. Actual backend/settings cannot be certified without a tag.
            if (role == "ordinary" and expected["backend"] == "ccgdf"
                    and getattr(mydf, "allow_legacy_df_cache", False)
                    and "j2c/metric_factors" not in f):
                logging.warning(
                    "Reusing explicitly allowed untagged legacy CCGDF cache %s; "
                    "backend/settings provenance is unverifiable. Regenerate to obtain a checked cache.", path)
                return "legacy-unverified"
            _mismatch(path, "untagged cache; regenerate it (legacy CCGDF requires --allow_legacy_df_cache true)")
        group = f[GROUP]
        if group.attrs.get("schema_version") != SCHEMA_VERSION or "request" not in group:
            _mismatch(path, "unsupported or incomplete provenance schema")
        request_text = group["request"].asstr()[()]
        if group.attrs.get("request_sha256") != hashlib.sha256(request_text.encode()).hexdigest():
            _mismatch(path, "damaged provenance checksum")
        try:
            actual = json.loads(request_text)
        except (ValueError, TypeError):
            _mismatch(path, "invalid provenance JSON")
        changed = [key for key in sorted(set(actual) | set(expected))
                   if actual.get(key) != expected.get(key)]
        if changed:
            _mismatch(path, "changed " + ", ".join(changed))
        producer = group.attrs.get("producer")
        required_producer = ("green_igen.df._make_j3c" if role == "ewald" else
                             "_RSGDFBuilder" if expected["backend"] == "rsgdf" else "_CCGDFBuilder")
        if producer != required_producer:
            _mismatch(path, "actual producer does not match the interaction/backend")
        if role == "ordinary" and expected["backend"] == "rsgdf":
            if "j2c/metric_factors" not in f or "j2c/qmesh" not in f:
                _mismatch(path, "missing actual RSGDF frame metadata")
            factors = f["j2c/metric_factors"]
            nq = len(f["j2c/qmesh"])
            if set(factors) != {str(i) for i in range(nq)}:
                _mismatch(path, "incomplete actual RSGDF frame metadata")
            naux = expected["auxiliary"]["naux"]
            for factor in factors.values():
                if factor.ndim != 2 or factor.shape[1] != naux or not 0 < factor.shape[0] <= naux:
                    _mismatch(path, "invalid actual RSGDF frame dimensions")
    logging.info("Reusing checked %s DF cache %s (actual producer %s)", role, path, producer)
    return producer
