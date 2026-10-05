# Periodic RSGDF backend

`--df_backend rsgdf` explicitly selects the CPU PySCF range-separated Gaussian
DF builder. The default and `--df_backend ccgdf` retain CCGDF; there is no `auto`
selection. GREEN's CPU/GPU consumer choice remains `--kernel CPU|GPU`.

Ordinary integrals use `_RSGDFBuilder`. Corrected q=0 correlation blocks retain
`green_igen.df._make_j3c`, with the legacy Ewald Coulomb definition. All diagonal
blocks share that corrected producer's frame. A scoped compatibility context
restores both copied class hooks on normal exits and exceptions, including
inherited attribute ownership. Nested contexts in one thread are supported;
concurrent contexts are rejected. Unrelated legacy builds must not run concurrently
with this class-global hook.

HF exports explicitly use ordinary integrals. A retained `cderi_ewald.h5` does
not enable a correction implicitly. Correlation exports request that file
explicitly; missing or incompatible corrected blocks fail before existing
export files are replaced.

New ordinary and corrected internal CDERI files carry versioned provenance.
Reuse checks the backend, interaction role, actual producer, cell/AO and
auxiliary bases, k mesh, PySCF version, precision/FFT settings, decomposition,
rank threshold and symmetry settings. RSGDF reuse also requires complete actual
frame metadata. A mismatch raises an error before existing outputs are replaced;
start a new calculation directory rather than relabeling the cache.

Untagged caches are rejected by default. `--allow_legacy_df_cache true` explicitly
permits old ordinary CCGDF files without RSGDF frame metadata, with a warning
that their backend/settings provenance cannot be verified. It never permits
untagged RSGDF or corrected files. This compatibility option should be used only
for a legacy CCGDF cache whose calculation settings the user has checked.

RSGDF stores the actual metric whiteners from PySCF's q-group iterator, including
conjugation, rank cutoff and Cholesky/eigenvalue convention, in its internal CDERI
metadata. Stored auxiliary q transformations use those exact frames. Recomputing
eigenvectors independently can change signs or rotations within degenerate
subspaces even when reconstructed ERIs agree. This adapter preserves the existing
`input.h5`, `df_hf_int/` and `df_int/` consumer dataset layout, complex-double
packing, normalization and orbital ordering. CCGDF metadata behavior is unchanged.

| RSGDF producer route | Support |
|---|---|
| Ordinary integrals and default Ewald correction | Supported on tested cases |
| GF2/GW correlation using default Ewald files | Tested with existing consumers |
| `finite_size_kind gf2`, `gw`, `gw_s`, `coarse_grained` | Rejected with an explicit error |
| Long-range-only Coulomb operator | Rejected |
| Negative auxiliary metric | Rejected |
| GPU integral production / GPU4PySCF | Not implemented |
| GPU GF2 correlation | Not implemented; GPU selection dispatches GPU HF + CPU GF2 |

Validation uses PySCF 2.14.0 and unchanged GREEN 1.0.0 consumers. Backend tests
cover explicit/default dispatch, hook ownership/restoration, exceptions, nesting
and concurrency. Interaction tests compare fixed-q cross-pair ERIs and corrected
q=0 ERIs, and independently reconstruct polarization through stored spatial/TR
operators. Validation utilities and the CPU/GPU matrix are described in
[`examples/rsgdf_validation`](../examples/rsgdf_validation/README.md).

The default Si precision of 1e-8 has integral convergence residuals of a few
1e-8. A controlled 1e-10/1e-12 study passes the initial atol=1e-8, rtol=1e-7
comparison without widening it. Production precision is unchanged.

An independent check exposed a pre-existing mismatch between CCGDF's spatial
q operators and its actual auxiliary frame on Si. This branch corrects RSGDF's
frames and leaves the default CCGDF path intact. Spatially reduced CCGDF GW
results should not serve as the sole physical reference for RSGDF. The small
consumer matrix uses identical unreduced spatial inputs; Si frame validation
uses a direct full-q contraction.
