# GREEN Ge correlated validation utilities

Portable source for comparing CCGDF/RSGDF in GREEN GF2 CPU/hybrid and GW CPU/GPU. The hybrid uses GPU HF with CPU GF2 correlation. The compact test model retains the complete GTH-SZV valence basis with a GTH-PBE pseudopotential and a 2x2x2 k mesh. GREEN supplies the correlated energies; PySCF HF is initialization only.

This source-only branch contains no run outputs, workstation logs, evidence archives, scientific figures or presentation PDFs. Those deliverables remain local to the study. No pull request or merge is created.

## Pinned source

- Producer: green-mbtools 0f101cb945b4d13cb71c9926fa023942ec14b60f, in a separate checkout.
- Consumer: green-mbpt 2346c87763e20cea5c55b95ab769ee55421f7386.
- GPU extension: green-gpu 4dcb075a9084f801dfb04a513112a736f4423dbc.

Configure FIX_SRC, FIX_PYTHON, STUDY_HOME, STUDY_RUNS and GREEN_GRID_FILE before sourcing scripts/environment.sh. Place these scripts under STUDY_HOME/scripts. Use workstation resources and local scratch storage for large arrays. Restore the dependency revisions from the relevant build record when reproducing an existing study.

Build the isolated main executable with build-main.sh and the optional fixed-mu diagnostic with build_fixed_mu.py. Generate each volume with prepare_inputs.py; it calls the production exporter, checks interactions, migrates copies with the official utility, and freezes identical starting one-body/HF arrays. Inputs preserve incomplete attempts rather than silently overwriting them.

Establish converged central GW/GF2 states with run_green.py, then provide their HDF5 paths through INITIAL_GW_REFERENCE and INITIAL_GF2_REFERENCE. run_correlated_scan.py continues refined references in volume and checks matched common-mu outputs. The diagnostic reuses unchanged compiled GREEN libraries. run_stock_scan.py checks the same physical outputs through the unmodified standard executable and its normal particle-number search. Warm continuations are not independent cold SCF solutions.

reference-controls.json records default controls and a larger bounded allowance for a slow expanded GW case. qualify_window.py adds only the declared common boundary points if the initial grid does not bracket all minima. The four-rank optional reference layout requires the central consistency check in check-gf2-mpi.sh. Primary GF2 CPU/hybrid runs retain one rank.

analyze_eos.py --stage stock-validation fits primary curves and rejects an incomplete or failed matrix. Local seven-point fits use identical volumes across a method's backend/device curves, retain all accepted raw points, and keep broad-window fits as diagnostics. Report B_E from internal energy, not an uncomputed Helmholtz isothermal modulus. This small model does not establish experimental accuracy.

collect_evidence.py and verify_evidence.py create and verify a local evidence snapshot. plot_results.py renders an accepted summary; use --runs for the actual data root or omit it for an extracted evidence tree. These utilities perform no external upload.
