# LEGO: Latent-space Exploration for Geometry-aware Optimization of Humanoid Kinematic Design

Official code for **LEGO** (ICRA 2026): [arXiv:2604.08636](https://arxiv.org/abs/2604.08636).

LEGO encodes the kinematic structures of 30 existing humanoid robots into a
compact latent space and searches that space for an upper-body kinematic design
that best reproduces a target human motion.

```
30 humanoid MJCFs ──► screw-axis representation ──► Isometric AutoEncoder ──► latent space
                                                                                  │
human motion (BVH / SMPL) ──► retarget + Procrustes objective ◄── VOO search ◄─────┘
```

---

## Repository layout

```
LEGO_code/
├── package/                  # library code (imported from notebooks via sys.path)
│   ├── yoon/                 #   clustering_robots.py (RobotCharacteristics), ae.py (AutoEncoder)
│   ├── kinematics/           #   screw / DH representation, dataset builder
│   ├── motion_retarget/      #   directional pre-retargeting, per-robot JOI definitions (joi/)
│   ├── mujoco_helper/        #   MuJoCo wrapper (MuJoCoParserClass)
│   ├── kinematics_helper/    #   FK / IK, kinematic chains, rotation utilities
│   ├── irvae_tools/          #   isometric regularizer
│   ├── bvh_helper/, smpl/    #   motion loaders
│   └── utility/
├── asset/                    # git submodule → LEGO-assets (robot MJCFs, robot registry)
├── data/mjcf_test/           # pre-computed screw / DH arrays of the 30 robots (AE training data)
├── result/                   # pretrained AutoEncoders (h64_32_z{d}_...)
├── bvh/                      # the three CMU motions used in the paper (easy / middle / hard)
├── notebook/
│   ├── tutorial/             # ★ start here: step-by-step tutorial (00 to 05)
│   └── ae_test/              # full experiment notebooks + VOO result summaries
└── Parameter.md              # all hyper-parameters (AE, VOO, retargeting)
```

## Installation

The robot models live in a separate repository, [LEGO-assets](https://github.com/EndeavoringYoon/LEGO-assets),
included as a submodule at `asset/`:

```bash
git clone --recursive https://github.com/EndeavoringYoon/LEGO.git
# already cloned without --recursive?  git submodule update --init
cd LEGO

conda create -n lego python=3.10 -y
conda activate lego
pip install -r requirements.txt
```

`requirements.txt` covers the tutorials and the paper experiments. A few helpers in
`package/` need extra packages that are imported only when those helpers are called:

```bash
pip install cvxpy     # smooth_optm_1d() in package/utility/utils.py
pip install shapely   # 2D polygon helpers in package/utility/utils.py
```

Tutorials 04 and 05 render offscreen with `mujoco.Renderer`, so they open no window. On a
headless machine, either select a headless OpenGL backend or run Jupyter under a virtual display:

```bash
sudo apt install libosmesa6 && MUJOCO_GL=osmesa jupyter lab   # software rendering, no X server
MUJOCO_GL=egl jupyter lab                                     # GPU rendering, if your driver provides EGL
sudo apt install xvfb && xvfb-run -a jupyter lab              # virtual display
```

The interactive MuJoCo viewer (`init_viewer`) and the slider tools used by some notebooks in
`notebook/ae_test/` open real windows and need a display (or `xvfb-run`).

The viewer window runs in a separate process (`package/mujoco_helper/render_process.py`): it keeps
rendering at `render_hz` (default 60) while the notebook computes, so slow loops such as
`explore_latent_dimension()` or the IK debug view (`animate=True`) do not drop frames and the camera
stays responsive. `render()` only sends the state; `render_throttle=False` lets a fast loop run at
full speed instead of at most `render_hz`. Pass `init_viewer(render_process=False)` for the
in-process viewer (needed for RGB overlays, figures, mouse picking and perturbation).
In a plain `.py` script, call `init_viewer` under `if __name__ == '__main__':`; the viewer process is
started with `spawn`, which re-imports the script (notebooks need nothing).

## Quick start

```bash
cd LEGO
jupyter lab notebook/tutorial/
```

| Notebook | What it does | Runtime (CPU) |
|---|---|---|
| `00_data_preprocessing.ipynb` | 30 MJCF robots → `(30, 30, 6)` screw / `(30, 32, 5)` DH arrays | ~5 min |
| `01_latent_space_learning.ipynb` | Isometric AutoEncoder training (or load the pretrained model) | ~2 min |
| `02_design_optimization_voo.ipynb` | VOO search over the latent space for one motion (`DEMO_MOTION`, default `easy`) | ~5 min (easy) to 25 min (middle) |
| `03_analysis_visualization.ipynb` | Table I, latent-space map (Fig. 3) and best design per motion from the saved VOO summaries | ~1 min |
| `04_latent_traversal.ipynb` | Render robots morphing along a latent path | ~1 min |
| `05_retarget_to_designed_robot.ipynb` | Decode an optimized design, export it as MJCF, retarget a human motion (or your own BVH) onto it and render it | ~1 min |

Every tutorial is read-only by default (`SAVE_OUTPUTS = False`); see
[`notebook/tutorial/README.md`](notebook/tutorial/README.md) for details.

The full experiments used in the paper live in `notebook/ae_test/`:

| Notebook | Purpose |
|---|---|
| `data_creation.ipynb` | Dataset curation from the 30 MJCFs |
| `ae_final.ipynb` | AutoEncoder training sweep (latent dim, isometric regularizer) |
| `experiment1_real.ipynb` | Per-motion VOO experiments (easy / middle / hard) |
| `experiment2.ipynb` | Motion-set VOO experiments |
| `mr_lego_{easy,middle,hard}_with_postrigging.ipynb` | Retargeting the target motions onto the optimized LEGO designs |

**Re-running is safe.** The pretrained checkpoints in `result/` and the paper's VOO
results in `notebook/ae_test/{easy,middle,hard}_motion_*/`, `all_*/` are never
overwritten:

- `run_experiment()` (`package/yoon/ae.py`) saves to `save_root` (default `result/`)
  and, if a model with the same name already exists, appends a timestamp to the new
  model's name (pass `overwrite=True` to replace it deliberately). Tutorial 01 saves
  newly trained models to `notebook/tutorial/outputs/models/`.
- The experiment notebooks write their VOO outputs to `notebook/ae_test/runs/`
  (`OUT_ROOT`), which is gitignored.

## Data

| Data | Included | How to obtain |
|---|---|---|
| Robot kinematic arrays (`data/mjcf_test/`) | ✅ | - |
| Pretrained AutoEncoders (`result/`) | ✅ | - |
| CMU motions `141_16`, `18_15`, `79_02` (`bvh/`) | ✅ | Other CMU motions: BVH conversion at [una-dinosauria/cmu-mocap](https://github.com/una-dinosauria/cmu-mocap); copy `data/<subject>/<take>.bvh` to `bvh/cmu_mocap_full_dataset/cmu_bvh/data/` |
| SMPL body model (`data/smpl_model/model.pkl`) | ❌ (license) | Register at <https://smpl.is.tue.mpg.de/>, download the model, convert it with [`preprocess.py`](https://github.com/CalciferZh/SMPL) (`python preprocess.py <model>.pkl`, needs `chumpy`) and place the output at `data/smpl_model/model.pkl` |
| MDM-generated motions (`data/mdm_motion/`) | ❌ (size) | Generate with [MDM](https://github.com/GuyTevet/motion-diffusion-model) and export SMPL parameters with its `visualize.render_mesh` (`*_smpl_params.npy`) |

The paper uses the three CMU motions only. SMPL motions are an optional input: with the SMPL model
and a `*_smpl_params.npy` file in place, set `MOTION_SOURCE = 'smpl'` in tutorial 02 (design
optimization) or 05 (retargeting); tutorial 05 §7 walks through it. In your own code, pass
`motion_type='smpl'`, `pkl_path='../../data/smpl_model/model.pkl'` and `file_path=<the .npy>` to
`evaluate()` / `run_once()` in `notebook/tutorial/shared/voo_helpers.py`.

## Robot models

The MuJoCo models of the 30 robots are kept in a separate repository,
[LEGO-assets](https://github.com/EndeavoringYoon/LEGO-assets), because each model keeps its own
upstream license (several are non-commercial or share-alike). It is checked out at `asset/`.

Six of the 30 models (Alex, JAXON, Pepper, Poppy Humanoid, Romeo, AgiBot X1) are not yet
redistributed while we wait for permission from their authors; the LEGO-assets README lists
where to download them and the expected file paths. `data/mjcf_test/` already contains the
pre-computed kinematic arrays of all 30 robots, so **tutorials 01-05 and the VOO experiments run
without them**: only `00_data_preprocessing.ipynb` / `data_creation.ipynb` need them.

## License

The code in this repository is released under the [Clear BSD License](LICENSE)
(BSD-3-Clause-Clear), Copyright (c) 2026 Korea University Research and Business Foundation.
This license grants no patent rights. Patent applications covering the method are pending in Korea
and the US; for commercial licensing, please contact the Korea University Research and Business
Foundation.

The license does **not** apply to the robot models in `asset/` (LEGO-assets), which are governed by
their own upstream licenses.

The three CMU motions in `bvh/` are from the
[CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/), funded by NSF EIA-0196217,
in the BVH conversion by Bruce Hahne (cgspeed.com).

Code adapted from IRVAE, CalciferZh/SMPL and OpenAI Gym keeps its original license; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{lego2026,
  title     = {LEGO: Latent-space Exploration for Geometry-aware Optimization of Humanoid Kinematic Design},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
  eprint    = {2604.08636},
  archivePrefix = {arXiv}
}
```
