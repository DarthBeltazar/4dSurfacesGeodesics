# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Greenfield. The only real content is `TASK.md` (in Russian) — the full task spec for a renderer of curved 3D
spaces (wormholes etc.) defined as hypersurfaces `M = {x ∈ R⁴ : f(x) = 0}`, with light rays as geodesics of the
induced metric. **Do not edit `TASK.md` without the user's consent.** There is no source code and no
`CMakeLists.txt` yet (`cmake-build-debug/` and `.idea/` are CLion artifacts, git-ignored). No build/test commands
exist yet — once they do, record them here and in `README.md`.

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
