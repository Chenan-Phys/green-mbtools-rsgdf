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
rank threshold and symmetry settings. Ordinary 3D reuse also requires the metric
algorithm and complete actual frame/dual metadata for both builders. A mismatch
raises an error before existing outputs are replaced;
start a new calculation directory rather than relabeling the cache.

Untagged caches are rejected by default. `--allow_legacy_df_cache true` explicitly
permits old ordinary 3D Cholesky CCGDF files without frame metadata, with a warning
that their backend/settings provenance cannot be verified. It never permits
untagged RSGDF, 3D eigenfactorized or corrected files. This option should be used only
for a legacy CCGDF cache whose calculation settings the user has checked.

Both 3D builders store the actual metric whiteners from the q-group iterator, including
conjugation, rank cutoff and Cholesky/eigenvalue convention, in its internal CDERI
metadata. Stored auxiliary q transformations use those exact frames. Recomputing
eigenvectors independently can change signs or rotations within degenerate
subspaces even when reconstructed ERIs agree. This adapter preserves the existing
`input.h5`, `df_hf_int/` and `df_int/` consumer dataset layout, complex-double
packing, normalization and orbital ordering. Internal CCGDF frame metadata now
follows the same convention as RSGDF.

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
1e-8. Cross-backend fixture comparisons pass at 1e-10; the stricter independent
full-q polarization check for both producers requires the controlled 1e-12
validation inputs. The CCGDF nonzero-q residual falls from 3.53e-8 at 1e-10 to
1.31e-9 at 1e-12, without widening atol=1e-8, rtol=1e-7. Production precision
is unchanged.

An independent check exposed a pre-existing mismatch between CCGDF's spatial
q operators and its actual auxiliary frame on Si. The earlier implementation
preserved that CCGDF metadata path. New three-dimensional builds now retain
actual frames and stable duals for both producers. Use fresh caches and the
independent full-q polarization check for both; old spatially reduced CCGDF
files are not a reference for the new convention. The small consumer matrix
uses identical unreduced spatial inputs.

## All-electron metric stability

Three-dimensional ordinary CCGDF and RSGDF builds share the same metric policy.
The normal CLI still selects Cholesky. Explicit eigenfactorization, and the
eigen fallback after failed Cholesky, use an equilibrated Cholesky/SVD route.
Write `J = D S D`, with `D = sqrt(diag(J))`, and factor `S = L L†`. The SVD
of `A = D L` gives `A = U sigma V†`; hence the original metric's eigenvalues
are `sigma^2`. Compute `W = V† L^-1 D^-1` and `R = D L V` on the retained
rows/columns, using triangular solves. These are the eigen whitener and its
stable dual, without inverting inaccurate eigenvalues from a direct `eigh(J)`.
They satisfy `W J W† = I` and `W R = I` in the retained space.

The cutoff preserves PySCF's original absolute raw-metric units:
`sigma^2 > linear_dep_threshold`. Scaling is a numerical preconditioner, not
a new rule for dropping auxiliary modes. This preserves the intended raw
eigenspace under crystal rotations. A semidefinite metric uses its numerically
resolved scaled positive subspace before the same raw-spectrum SVD cutoff.
Invalid 3D positive metrics fail explicitly. Low-dimensional negative-metric
support is not extended by this change.

Both builders store their actual whiteners, duals and factorization tags.
Auxiliary symmetry uses these duals instead of recomputing a raw eigensystem or
an ill-scaled pseudoinverse. Ordinary cache provenance includes the metric
algorithm and frame version 2, so old unscaled caches require regeneration.
The legacy Ewald producer and its cache policy are unchanged. Untagged legacy
CCGDF eigen caches cannot certify this convention and are refused even with
the legacy compatibility option; the explicit old Cholesky compatibility
remains available.

This corrects the all-electron Ge eigenfactorization failure: raw eigenmetrics
with condition numbers around 1e16 can lose their smallest modes to roundoff.
Raw factor arrays may still differ by auxiliary gauge rotations. Compare
fixed-q cross-pair Coulomb contractions, not factor entries across builders.
