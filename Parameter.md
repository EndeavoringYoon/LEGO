# Parameter.md

Parameters used in the Autoencoder and VOO (Voronoi Optimistic Optimization) experiments.
Reference files: `notebook/ae_test/ae_final.ipynb`, `notebook/ae_test/experiment2.ipynb`

---

## 1. AutoEncoder (`AutoEncoderClass`)

### Architecture parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x_dim` | `int` or `tuple` | - | Input dimension. `int` for a 1D vector, `tuple` for a multi-dimensional tensor (flattened internally). |
| `z_dim` | `int` | `2` | Latent space dimension. Swept over 2~120 in the experiments. |
| `h_dims` | `List[int]` | `[128, 128]` | Size of each encoder/decoder hidden layer. The decoder is symmetric, in reversed order. |
| `actv` | `nn.Module` | `nn.ReLU()` | Hidden-layer activation. `nn.Tanh()` is used in the experiments. |

### Loss function parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_type` | `str` | `'l2'` | Reconstruction loss type: `'l2'` (MSE), `'l1'` or `'huber'`. |
| `iso` | `bool` | `False` | Whether to use isometric regularization. If `True`, adds the isometric loss weighted by `iso_reg`. |
| `iso_reg` | `float` | `1e-7` | Weight λ of the isometric loss. Total loss = reconstruction loss + `iso_reg` × isometric loss. |
| `eta` | `float` | `0.2` | Perturbation size inside `relaxed_distortion_measure`. Applied only when `iso=True`. (`package/irvae_tools/geometry.py`) |

> **Total loss**: `L = L_recon + iso_reg × L_iso`
> The isometric loss regularizes the decoder to preserve the ratio between distances in the latent space and in the original space.

---

## 2. Training parameters (`run_experiment`)

| Parameter | Type | Default | Experiment value | Description |
|---|---|---|---|---|
| `h_dims` | `List[int]` | `[64, 32]` | `[64, 32]` | List of hidden-layer sizes. |
| `z_dim` | `int` | `2` | `2`~`120` | Latent dimension. Recorded as `z{z_dim}` in the `result/` folder name. |
| `num_epochs` | `int` | `1000` | `1000` | Number of training epochs. |
| `lr` | `float` | `1e-3` | `1e-3` | Adam learning rate. |
| `batch_size` | `int` | `1` | `10` | Mini-batch size. |
| `device` | `str` | `'cuda:0'` | `'cuda:0'` | Training device. |
| `actv` | `nn.Module` | `nn.Tanh()` | `nn.Tanh()` | Activation function. |
| `loss_type` | `str` | `'l2'` | `'l2'` | Reconstruction loss type. |
| `iso` | `bool` | `False` | `True` | Whether to use isometric regularization. |
| `iso_reg` | `float` | `1e-7` | `1e-7` | Isometric loss weight. |

### Model save path and naming convention

Results are saved under `result/`, with folder names of the form:

```
h{h_dims}_z{z_dim}_iso{iso}_actv{actv_name}_isoreg{iso_reg}_nepoch{num_epochs}_bs{batch_size}
```

**Example**: `h64_32_z2_isoTrue_actvTanh_isoreg1e-07_nepoch1000_bs10`

Files saved in each folder:
- `model.pt`: model weights (state_dict)
- `config.json`: records all parameters above

---

## 3. Input data

| Item | Value | Description |
|---|---|---|
| `repr_kind` | `'screw'` or `'dh'` | Kinematic representation: `screw` (screw theory) or `dh` (DH parameters). |
| `data_path` | `data/mjcf_test/{repr_kind}/` | Training data path. |
| Data shape | `(30, 20, 6)` -> flatten `(30, 120)` | 30 robots, 20 joint groups, 6 features (for screw: 3 for w + 3 for q). |
| Joint indices used | `[0..9, 20..29]` (screw) | Only the 20 arm (upper-limb) groups out of all 30. |

---

## 4. VOO (Voronoi-based Optimistic Optimization)

Black-box optimization algorithm that searches the latent space for the optimal robot design.

### Search space parameters

| Parameter | `ae_final.ipynb` | `experiment2.ipynb` | Description |
|---|---|---|---|
| `x_dim` | `ae.z_dim` | `ae.z_dim` | Search space dimension = latent dimension. |
| `x_minval` | `-10` | `-15` | Lower bound of the latent search space. |
| `x_maxval` | `10` | `15` | Upper bound of the latent search space. |

### VOO algorithm parameters

| Parameter | `ae_final.ipynb` | `experiment2.ipynb` | Description |
|---|---|---|---|
| `n_init` | `5` | `16` | Number of initial random samples. |
| `max_iter` | `10` | `100` (latent) / `30` (raw) | Number of optimization iterations. |
| `p_global` | `0.5` | varies | Global exploration probability. Below `p_global`: global sample, otherwise local search. |
| `invlen` | `1.0` | varies | Scale of the Voronoi distance. Passed as the `invlen` argument of `r_sq`. Larger values shrink the neighborhood radius. |
| `n_loop_switch` | `10` | varies | Loop count after which local search switches from global tries to local proposals. |
| `sigma_coef` | unused | varies | Std scaling coefficient of the local proposal distribution. `σ = sigma_coef × √(dist_min / x_dim)`. |
| `max_loop` | `1000` | varies | Maximum iterations of the local-search while loop. Exits with a global sample when exceeded. |

### VOO algorithm flow

```
Initialization: x_voo (n_init random samples) -> evaluate y_voo

for it in range(max_iter):
    if rand < p_global:
        x_sel <- global random sample in [x_minval, x_maxval]
    else:
        x_min_pt <- coordinates of argmin(y_voo)
        dist_min <- minimum Voronoi distance between x_min_pt and the other points

        while True:
            if n_loop < n_loop_switch:
                x_sel <- global random sample  (search for a point inside the Voronoi cell)
            else:
                σ = sigma_coef × √(dist_min / x_dim)
                x_sel <- x_min_pt + σ·N(0,I)  (local Gaussian proposal)
                x_sel ← clip(x_sel, x_minval, x_maxval)

            if r_sq(x_min_pt, x_sel) < dist_min: break
            if n_loop >= max_loop: x_sel <- global random; break

    y_sel ← evaluate(x_sel)
    x_voo, y_voo ← append(x_sel, y_sel)
```

### Evaluation function parameters (`evaluate`)

| Parameter | Default | Description |
|---|---|---|
| `eval` | `'procrustes'` | Evaluation metric. Procrustes distance (motion retargeting error). |
| `jnt_penalty` | `True` | Whether to include the joint-count penalty. Adds a penalty per joint. |
| `jnt_coeff` | `1` | Joint-count penalty coefficient. |
| `ik_p_spine` | `False` | Whether to include spine position in the IK. |
| `ik_p_neck` | `True` | Whether to include neck position in the IK. |
| `leg_fixed` | `True` | Whether the legs are fixed. If `True`, lower-body IK is disabled. |
| `ik_R_hand` | `False` | Whether to include hand rotation in the IK. |

> **Evaluation score**: sum of Procrustes distances. Lower means the robot reproduces the original human motion better. Failed/empty robots get `10.0`.

---

## 5. Result folder layout (`result/`)

```
result/
└── h64_32_z2_isoTrue_actvTanh_isoreg1e-07_nepoch1000_bs10/
    ├── model.pt        # model weights
    └── config.json     # full training config
```

VOO search results (x, y sequences) are saved separately as `.npy`:
```
{model_tag}VOO.npy   # shape: (n_init + max_iter,)  best-so-far y sequence
```
