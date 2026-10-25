# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`TASK.md` (in Russian) is the full task spec for a renderer of curved 3D spaces (wormholes etc.) defined as
hypersurfaces `M = {x ∈ R⁴ : f(x) = 0}`, with light rays as geodesics of the induced metric. **Do not edit
`TASK.md` without the user's consent.** Stages 1–5 are done (Python + JAX prototype, mode A, GPU port in `gpu/`,
path tracing, interactive app); `cmake-build-debug/` and `.idea/` are CLion artifacts, git-ignored (`.run/` too: generated locally). Read `docs/PROGRESS.md` for the current state.

## Commands (Python prototype, venv at `.venv/`; on Linux/macOS use `.venv/bin/python`)

```sh
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q                                   # full suite, ~15 min
.venv/Scripts/python -m pytest tests/test_u1_jet.py -q                    # one file
.venv/Scripts/python -m pytest "tests/test_u2_reference.py::test_vs_dop853[ellis]"   # one case
PYTHONIOENCODING=utf-8 .venv/Scripts/python prototype/scripts/<name>.py   # experiments → results/stage-1/
```

GPU (Vulkan SDK 1.4.328.1 with slangc/volk/VMA, MSVC 2022; CMake/Ninja are taken from CLion):

```sh
gpu\build.cmd                                         # → gpu/build/geo4d_gpu.exe (info | run | graph | denoise | app | export-graph)
gpu/build/geo4d_gpu.exe scenes/two_wormholes.json --res 1280x720   # окно без Python (DD-010); app.py — обёртка
./gpu/build.sh                                        # Linux/macOS (ADR-0008); deps via FetchContent if no SDK
.venv/Scripts/python -m pytest tests -q -m gpu       # GPU tests, ~5 min; skipped if the exe or slangc is missing
```

`pyproject.toml` puts `prototype/` on the pytest path. Experiments write `results/stage-N/<date>_<name>/`
via `geo4d.results.new_run`; commit code before runs whose numbers go into docs (meta.json records the git
hash and a dirty flag).

## Code architecture (`prototype/geo4d/`)

- SDFs are written once against `jet.ops` + arithmetic and run on jnp arrays (reference autodiff), `Jet7`
  (f, ∇f, ∇f·v, vᵀHv in one pass) and `Jet15` (full Hessian). Never use hard min/max/abs in geometry; branch
  selection via `ops.where` is allowed only when both branches are the same smooth function.
- `shapes.Shape` bundles f, scale ℓ_s, α (= 0.1/ℓ_s, ADR-0001), sampler, conserved planes, exact solution and
  `hcap` (step cap). Any scene with flat zones **must** set `hcap` (sphere tracing to the curvature support),
  otherwise adaptive steps tunnel through curvature.
- `integrators.make_integrator(shape, Config, escape=, aux=)` returns a jitted vmap over a per-ray
  `lax.while_loop`; it counts SDF passes per ray (`nsdf`), the benchmark metric. Projection (M3) reuses the
  FSAL stage's jet; FSAL "stale" is the default.
- `reference.py` is independent of jet code (DOP853 on `jax.grad`/`jax.hessian`, quadratures) — keep it so.
- Jacobi fields: `Config(jacobi=k)` adds a channel E = (J, J′) of shape (2, k, 4); every RK stage then does one
  Jet15 pass (counted as `nsdf15`). J′ is the covariant derivative; ambient ∂v = J′ − ⟨SJ,v⟩n (M17) — use
  `transfer.state_derivatives`, never raw J′, when differentiating anything of (x, v).
- Mode A (`modea.py`) and its reference/fallback must share the cubemap, the mip-0 bilinear sampler and the filter
  model (Gaussian pixel σ = 0.5 px, Gaussian aperture); PSNR is on tonemapped values. Use
  `cubemap.sample_gauss_exact` (dense grid), not low-order Gauss–Hermite, as the "ideal filter".
- In tests, never draw direction hints from the same RNG stream/seed as `shapes.sample` (on S³ they coincide with
  the normal); `make_camera` falls back robustly, but test axes must be decorrelated.
- `x64` must be enabled before any array is created (`geo4d/__init__.py`, `tests/conftest.py`).

## GPU architecture (`gpu/`, DD-004)

- Shaders are compiled by Python (`geo4d.gpu.compile_shader`, cache `gpu/build/spv/`, scene chosen by define), not
  by CMake; rebuilding `gpu\build.cmd` is needed only for host changes. Host = generic executor: `run` (one
  dispatch from a job file) and `graph` (named buffers, indirect dispatches, loops, timestamps); scenes, cameras
  and result parsing live in `geo4d.gpu` / `geo4d.gpu_modea`, data via `.npy`.
- SDFs are written once in `scenes.slang` against `IReal` (`jet.slang`: Real, Jet7, JetHVP, Jet15, Dual4); keep the
  scene IDs in sync with `gpu.SCENE_IDS`. Push constants must be plain Python floats (np.float64 repr broke parsing).
- GPU correctness criterion: error vs DOP853 ≤ 4× that of the float32 prototype (`GEO4D_X64=0`) on the same inputs.
- Float32 defaults: κ = 3e-3 (ADR-0004); background SampleGrad k = 2 + BC6H, M19 optional (ADR-0005).

## Mode B (stage 4, DD-005)

- Scenes are defined once in `pathtrace.py` (objects carry a `spec` for codegen); `gpu_pt.scene_header` generates
  `pt_scene.slang` into `gpu/build/gen/<hash>/` (passed via `-I`). Add a new object kind in both `objects.py` and `_expr`.
- Hits are detected only by a sign change of g (or g below the rounding floor) — never by `g < ε_hit`: near misses
  within ε are not hits (this bug cost 8 % energy at silhouettes). Same rule in shadow rays.
- NEE/MIS only from flat zones over straight segments on the same sheet (M29); weights must use exactly the same
  domain test on the BSDF side. Verify any change with the furnace and MIS-vs-NEE-only tests.
- Statistical comparisons against the prototype: per-pixel z is unreliable at a few hundred spp (heavy tails of
  small lights / BSDF-only Ellis); use the image-mean z plus a loose outlier fraction.
- Host accumulation: `pt_sample` adds one path per pixel per dispatch (`repeat`), state lives in inout buffers;
  `geo4d_gpu denoise` runs OptiX via `nvcuda.dll` + a `cuda.h` shim (ADR-0007).

## Interactive app (stage 5, DD-007)

- `geo4d_gpu app <graph>` runs graph sections per frame: `move` (cam_step → camera pass → mode A → lists → reset → PT),
  `static` (+1 spp), `post` (present); app directives (`app`, `denoise`, `campipe`, `record`, `dump`, `bench`) are in
  the graph file, generated by `gpu_pt.app_graph`. Push indices for runtime updates: `gpu_pt.I_SEED/I_USE_DEN/I_BGRA`.
- Denoiser buffers are exported Vulkan memory imported into CUDA (`buffer X export`); keep their layout (H, W) float4.
- Scenes without background (compact spaces) skip the camera pass (`pt_classall`).
- Edit-mode graph (DD-008/DD-009) is fully dynamic: objects *and* manifold come from rows of the `cam` buffer
  (`SCENE_DYNAMIC`, rows at `gpu_pt.MF_BASE`, layout in `scenes.slang`); it has `move/edit` and `move_nobg/edit_nobg`
  sections, the host picks by the background flag. Keep `gpu_pt.manifold_rows` and `editor.cpp::manifold_info` in sync
  (`mf_check`, like `rows_check`). Opening a scene / changing the manifold applies instantly there; leaving edit mode
  regenerates the fast graph when structure or manifold key differ.
- Render resolution ≠ window size is fine (blit with letterbox); a resolution change is a Python rebuild with a new
  `--res`. `app.py` caches the procedural background on disk (`procedural_env_cached`) — rebuilds rely on it.

## Build layout, platforms, IDE (ADR-0008)

- The root `CMakeLists.txt` just does `add_subdirectory(gpu)`; CLion opens the root. `CMakePresets.json`: `release` →
  `gpu/build` (Python's `geo4d.gpu.EXE`), `debug` → `gpu/build-debug`. Override the exe with `GEO4D_GPU_EXE` and
  slangc with `GEO4D_SLANGC`.
- External memory is per-platform (Win32 HANDLE / Linux fd; after a CUDA import the fd belongs to CUDA). macOS has
  no external memory or OptiX. The denoiser is optional at runtime: if it fails, the app runs without it. Keep
  Windows-only APIs behind `_WIN32`.
- Run configs in `.run/` (not in git) are generated by `prototype/scripts/ide_configs.py` (edit the script, not the XML).
  Python configs need `.venv` registered as an interpreter in the IDE. `app.py --graph-only DIR` writes a
  self-contained graph so the host can run under the CLion debugger.
- Full-size frame dumps (`Frame_/Den_/AlbF_/NrmF_/Present_/swapchain_*.npy`) are git-ignored; commit PNG previews.
- Linux is verified only in WSL2 on lavapipe without the denoiser; macOS is unverified.

## Planned stages (gate: don't start stage N+1 until stage N acceptance criteria pass)

1. **Python + JAX prototype**: jet numbers, SDFs for all scenes/test shapes, geodesic RHS, Bogacki–Shampine 3(2)
   with FSAL + adaptive step, Baumgarte stabilization, projection. Tests U1–U6, Ellis deflection vs quadrature,
   convergence order ≈3, SDF-calls-per-ray benchmark (fixed-step project, RK4, BS32, DOPRI5 × S³/ellipsoid/T³/Ellis).
2. **Mode A** (background only): Jacobi fields → 4×4 ray transfer matrix, thin lens, EWA/textureGrad background
   filtering, nonlinearity detector → stochastic fallback. U7 must pass before rendering.
3. **GPU port**: Slang → Vulkan compute (or CUDA C++ if justified), float32, BC6H cubemaps, wavefront queues.
4. **Mode B**: path tracing with 4D object SDFs intersected with M, denoiser.
5. Optional interactive camera / offline video.

Planned layout: `prototype/ gpu/ tests/ scenes/ results/` plus `docs/` (see below).

## Non-negotiable math/implementation rules from the spec

- Geodesic ODE (extrinsic form, state (x,v) ∈ R⁸): `ẍ = −(vᵀHv + 2α∇f·v + α²f)·∇f/|∇f|²`; after every accepted
  step project x back onto f=0 and v onto the tangent plane, renormalize v. |∇f| = 1 is not assumed.
- Derivatives come from **jet numbers, never finite differences**: 7-component jet `a + b·t + c·t² + Σdᵢsᵢ`
  (one SDF pass gives f, ∇f·v, vᵀHv = 2c, ∇f); a 15-component second-order jet (or equivalent) for the full
  Hessian in the Jacobi pass.
- **Hard min/max/abs are forbidden in geometry SDFs** — use smooth versions (smin, zero-guarded length).
- Integrator error is measured on tangential components only; primary-ray tolerance tied to pixel angular size;
  step also capped by object SDF g(x).
- Rays near unstable orbits are exponentially sensitive — physics, not a bug: cap steps and flag the pixels.
- Correctness before speed; every performance claim must be backed by a measurement. If a prescribed method
  loses to an alternative in practice, show measurements and propose a replacement — don't fudge results.
- Verify any new formula / sign convention numerically in the prototype first.

## Documentation & session workflow (required by the spec)

Work spans many sessions; everything needed to continue must live in the repo.

- **Session start**: read `TASK.md`, `docs/PROGRESS.md`, the latest `docs/log/` entry, and relevant DDs.
- **Session end**: rewrite (not append to) `docs/PROGRESS.md` (≤ ~1 screen: current stage, criteria checklist,
  next 3–5 steps, blockers, key numbers with links), add `docs/log/YYYY-MM-DD-NN.md` (including what failed and
  why), update affected DD/ADR/`docs/math.md`, commit.
- `docs/math.md` is the single source of formulas, numbered M1, M2, …; code references them
  (`// M7: Jacobi equation, see docs/math.md`). Each formula links to the test that verifies it.
- `docs/design/DD-NNN-*.md` are written **before** implementing a major subsystem (DD-001…DD-006, template in the
  spec). **DD-004 (GPU architecture) and DD-005 (path tracing): stop and request user review once in Review status,
  before implementing.**
- `docs/adr/ADR-NNNN-*.md` for small decisions; never delete, mark Superseded.
- `docs/reports/stage-N.md` on stage completion — then stop and give the user a short summary.
- Every number in docs must link to a `results/stage-N/YYYY-MM-DD_<name>/` dir containing `config.json`,
  `meta.json` (commit hash, GPU, driver, OS, lib versions), raw data, plots, and `cmd.txt`. Large binaries not in git.
- Commits: small, prefixed by stage/subsystem, e.g. `[s1/integrator] BS32 with FSAL`, `[docs] DD-003 Review`;
  tests and docs go in the same commit as the code they cover.
