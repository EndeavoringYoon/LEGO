"""Isometric autoencoder (IsoAE) used to learn the LEGO latent space."""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sys,os
import numpy as np
from geometry import relaxed_distortion_measure
from IPython.display import clear_output, display
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

class AutoEncoderClass(nn.Module):
    """Fully connected autoencoder with an optional isometric regularizer.

    Args:
        x_dim: input dimension (int), or input shape (tuple, flattened internally).
        z_dim: latent dimension.
        h_dims: hidden layer sizes of the encoder; the decoder mirrors them.
        loss_type: reconstruction loss, 'l2', 'l1' or 'huber'.
        actv: activation module used between layers.
        iso: add the isometric (relaxed distortion) loss.
        iso_reg: weight of the isometric loss.
    """
    def __init__(
        self,
        x_dim,
        z_dim=2,
        h_dims=[64,32],
        loss_type='l2',           # 'l2', 'l1', 'huber'
        actv=nn.ReLU(),           # activation function
        iso = True,
        iso_reg = 1e-7,
    ):
        super(AutoEncoderClass, self).__init__()
        self.name = 'autoencoder'
        self.z_dim = z_dim
        self.h_dims = h_dims
        self.loss_type = loss_type.lower()
        self.actv = actv
        self.iso = iso
        self.iso_reg = iso_reg

        if isinstance(x_dim, int):
            self.input_shape = (x_dim,)
            self.flattened_dim = x_dim
        elif isinstance(x_dim, tuple):
            self.input_shape = x_dim
            self.flattened_dim = int(np.prod(x_dim))
        else:
            raise ValueError("x_dim must be int or tuple")

        enc_layers = []
        in_dim = self.flattened_dim
        for h_dim in self.h_dims:
            enc_layers.append(nn.Linear(in_dim, h_dim))
            enc_layers.append(self.actv)
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, self.z_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = self.z_dim
        for h_dim in reversed(self.h_dims):
            dec_layers.append(nn.Linear(in_dim, h_dim))
            dec_layers.append(self.actv)
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, self.flattened_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        """Map x of shape (B, *input_shape) to z of shape (B, z_dim)."""
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z):
        """Map z of shape (B, z_dim) to x of shape (B, *input_shape)."""
        return self.decoder(z).view(z.size(0), *self.input_shape)

    def forward(self, x):
        """Return (reconstruction, z)."""
        z = self.encode(x)
        return self.decode(z), z
    
    def compute_loss(self, x, x_hat):
        """Return (total_loss, reconstruction_loss, isometric_loss) for a batch x and its reconstruction x_hat."""
        if self.loss_type == 'l2': 
            loss = nn.functional.mse_loss(x_hat, x)
        elif self.loss_type == 'l1': 
            loss = nn.functional.l1_loss(x_hat, x)
        elif self.loss_type == 'huber': 
            loss = nn.functional.huber_loss(x_hat, x)
        elif self.loss_type == 'mseloss':
            loss = nn.functional.mse_loss(x_hat,x)
        else: 
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        if self.iso:
            iso_loss = relaxed_distortion_measure(self.decode, self.encode(x), eta=0.2, metric = 'identity')
            total_loss = loss + self.iso_reg * iso_loss
        else:
            iso_loss = th.tensor(0.0, device=x.device, dtype=loss.dtype)
            total_loss =loss
        return total_loss, loss, iso_loss

def run_experiment(data_np,
        h_dims=[64,32],
        z_dim=2,
        num_epochs=1000,
        lr=1e-3,
        actv=nn.Tanh(),
        loss_type='l2',
        device='cuda:0',
        batch_size=1,
        iso=False,
        iso_reg=1e-7,
        model_name=None,
        save_root="../../result",
        overwrite=False,
        plot=True,
        ):
    """
    save_root: directory to save trained models to. Defaults to ../../result, same as the pretrained models.
    overwrite: if False and a model with the same name already exists, do not overwrite it; save with a timestamp appended to the name.
               (prevents the pretrained checkpoints from being overwritten)
    plot: if True, plot the loss curve every 10 epochs.
    """
    data_tensor = th.tensor(data_np, dtype=th.float32)
    dataset = th.utils.data.TensorDataset(data_tensor)
    loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x_dim = data_tensor.shape[1:]
    results = []
    if model_name is None:
        actv_name = actv.__class__.__name__
        h_str = "_".join(str(h) for h in h_dims)
        model_name = f"h{h_str}_z{z_dim}_iso{iso}_actv{actv_name}_isoreg{iso_reg}_nepoch{num_epochs}_bs{batch_size}"
    if not overwrite and os.path.exists(os.path.join(save_root, model_name, "model.pt")):
        old_name = model_name
        model_name = f"{model_name}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        print(f"[run_experiment] '{old_name}' already exists in {save_root}; saving as '{model_name}' instead.")

    ae = AutoEncoderClass(x_dim=x_dim, z_dim=z_dim, h_dims=h_dims, loss_type=loss_type, actv=actv,iso=iso,iso_reg=iso_reg).to(device)
    optim = th.optim.Adam(ae.parameters(), lr=lr)

    pbar = tqdm(range(num_epochs), desc='Training', leave=False)

    total_losses = []
    recon_losses = []
    iso_losses = []
    epochs = []

    for epoch in pbar:
        ae.train()
        total_total = 0.0
        total_recon = 0.0
        total_iso = 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _ = ae(batch)
            total_loss, recon_loss, iso_loss = ae.compute_loss(batch, x_hat)

            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(ae.parameters(), max_norm=float('inf')).item()
            optim.step()
            optim.zero_grad()

            total_total += total_loss.item() * batch.size(0)
            total_recon += recon_loss.item() * batch.size(0)
            total_iso += iso_loss.item() * batch.size(0)

        avg_total = total_total / len(dataset)
        avg_recon = total_recon / len(dataset)
        avg_iso = total_iso / len(dataset)

        total_losses.append(avg_total)
        recon_losses.append(avg_recon)
        iso_losses.append(avg_iso)
        epochs.append(epoch)

        pbar.set_description(f"epoch={epoch}, total={avg_total:.4f}, recon={avg_recon:.4f}, iso={avg_iso:.4f}")

        if plot and (epoch % 10 == 0 or epoch == num_epochs - 1):
            clear_output(wait=True)

            # Total Loss
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(epochs, total_losses, label="Total Loss", color='tab:blue')
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Total Loss")
            ax1.grid(True)
            ax1.legend()
            display(fig1)
            plt.close(fig1)

            # Reconstruction Loss
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(epochs, recon_losses, label="Reconstruction Loss", color='tab:orange')
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("Reconstruction Loss")
            ax2.grid(True)
            ax2.legend()
            display(fig2)
            plt.close(fig2)

            # Isometric Loss
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.plot(epochs, iso_losses, label="Isometric Loss", color='tab:green')
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Loss")
            ax3.set_title("Isometric Regularization Loss")
            ax3.grid(True)
            ax3.legend()
            display(fig3)
            plt.close(fig3)

    # ────── prepare save path ──────
    save_dir = os.path.join(save_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # ────── save model ──────
    model_path = os.path.join(save_dir, "model.pt")
    th.save(ae.state_dict(), model_path)

    # ────── save config ──────
    config = {
        "x_dim": x_dim,
        "h_dims": h_dims,
        "z_dim": z_dim,
        "num_epochs": num_epochs,
        "lr": lr,
        "loss_type": loss_type,
        "batch_size": batch_size,
        "iso": iso,
        "iso_reg": iso_reg,
        "device": device,
        "actv": actv.__class__.__name__,  # store the activation name
    }
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return results, model_name