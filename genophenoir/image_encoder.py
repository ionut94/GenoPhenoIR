"""Phase 4 — Visual Encoding Explorer.

Creates 2D image encodings of per-accession IR profiles (genomic position ×
features), trains a CNN autoencoder to learn latent representations, and
compares image-based clustering with the tabular approach from Phase 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from PIL import Image
from tqdm import tqdm

from genophenoir.config_loader import Config, load_config, resolve_path

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Image creation
# ---------------------------------------------------------------------------

def create_accession_images(
    ir_bed_path: Path,
    fingerprint_df: pd.DataFrame,
    phenotype_df: pd.DataFrame | None,
    chrom: str,
    chrom_length: int,
    window_size: int,
    out_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Create 2D image encodings for each accession.

    Each image has:
      X-axis: genomic position binned into windows
      Y-axis: feature rows (IR count, mean stem length, mean spacer length,
              SNP density in IRs, and optionally phenotype values)
      Pixel intensity: normalised feature value (0-255)

    Args:
        ir_bed_path: Path to the IR BED file from Phase 1.
        fingerprint_df: IR fingerprint matrix (accessions x IR regions).
        phenotype_df: Optional phenotype values for additional rows.
        chrom: Chromosome name.
        chrom_length: Length of the chromosome in bp.
        window_size: Genomic window size in bp.
        out_dir: Directory to save PNG images.

    Returns:
        Tuple of (image_array [n_accessions, height, width], accession_ids).
    """
    from genophenoir.ir_profiler import load_ir_bed

    # Load IR positions
    ir_list = load_ir_bed(ir_bed_path)
    ir_df = pd.DataFrame([{
        "ir_id": ir.ir_id,
        "start": ir.start,
        "end": ir.end,
        "stem_length": ir.stem_length,
        "spacer_length": ir.spacer_length,
        "midpoint": (ir.start + ir.end) // 2,
    } for ir in ir_list])

    # Create genomic windows
    n_windows = max(1, chrom_length // window_size)
    ir_df["window"] = np.clip(ir_df["midpoint"] // window_size, 0, n_windows - 1).astype(int)

    # Pre-compute per-window reference features
    window_stats = ir_df.groupby("window").agg(
        ir_count=("ir_id", "count"),
        mean_stem=("stem_length", "mean"),
        mean_spacer=("spacer_length", "mean"),
    ).reindex(range(n_windows), fill_value=0)

    # Map IR IDs to windows
    ir_id_to_window = dict(zip(ir_df["ir_id"], ir_df["window"]))

    accessions = list(fingerprint_df.index)
    n_base_features = 4  # ir_count, mean_stem, mean_spacer, snp_density
    n_pheno_features = len(phenotype_df.columns) if phenotype_df is not None else 0
    img_height = n_base_features + n_pheno_features
    img_width = n_windows

    images = np.zeros((len(accessions), img_height, img_width), dtype=np.float32)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, acc in enumerate(tqdm(accessions, desc="Creating images")):
        # Row 0: IR count per window (same for all accessions on reference)
        images[i, 0, :] = window_stats["ir_count"].values

        # Row 1: Mean stem length per window
        images[i, 1, :] = window_stats["mean_stem"].values

        # Row 2: Mean spacer length per window
        images[i, 2, :] = window_stats["mean_spacer"].values

        # Row 3: SNP density in IRs (proportion of disrupted IRs per window)
        if acc in fingerprint_df.index:
            acc_fp = fingerprint_df.loc[acc]
            disrupted_per_window = np.zeros(n_windows)
            total_per_window = np.zeros(n_windows)

            for ir_id, intact in acc_fp.items():
                w = ir_id_to_window.get(ir_id)
                if w is not None and 0 <= w < n_windows:
                    total_per_window[w] += 1
                    if intact == 0:
                        disrupted_per_window[w] += 1

            with np.errstate(divide="ignore", invalid="ignore"):
                snp_density = np.where(
                    total_per_window > 0,
                    disrupted_per_window / total_per_window,
                    0,
                )
            images[i, 3, :] = snp_density

        # Rows 4+: phenotype values (constant across windows for this accession)
        if phenotype_df is not None and acc in phenotype_df.index:
            for j, trait in enumerate(phenotype_df.columns):
                val = phenotype_df.loc[acc, trait]
                if pd.notna(val):
                    images[i, n_base_features + j, :] = val

    # Normalise each feature row to 0-255
    images_norm = np.zeros_like(images, dtype=np.uint8)
    for row in range(img_height):
        row_data = images[:, row, :]
        rmin, rmax = row_data.min(), row_data.max()
        if rmax > rmin:
            images_norm[:, row, :] = (
                (row_data - rmin) / (rmax - rmin) * 255
            ).astype(np.uint8)

    # Save individual PNGs
    for i, acc in enumerate(accessions):
        img = Image.fromarray(images_norm[i], mode="L")
        # Scale up for visibility
        img_scaled = img.resize((img_width * 2, img_height * 40), Image.NEAREST)
        img_scaled.save(img_dir / f"{acc}.png")

    logger.info(
        "Created %d images (%d x %d) saved to %s",
        len(accessions), img_height, img_width, img_dir,
    )

    return images_norm, accessions


# ---------------------------------------------------------------------------
# CNN Autoencoder (PyTorch)
# ---------------------------------------------------------------------------

def build_and_train_autoencoder(
    images: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    out_dir: Path | None = None,
) -> tuple[np.ndarray, object]:
    """Train a CNN autoencoder on the image encodings.

    Args:
        images: Array of shape (n_samples, height, width) with uint8 values.
        latent_dim: Dimensionality of the latent space.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        out_dir: Directory to save training loss plot.

    Returns:
        Tuple of (latent_vectors [n_samples, latent_dim], trained_model).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Prepare data: normalise to [0, 1], add channel dimension
    X = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)  # (N, 1, H, W)
    n_samples, _, img_h, img_w = X.shape

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Dynamic architecture based on image dimensions
    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.flat_size = 32 * img_h * img_w
            self.fc = nn.Linear(self.flat_size, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flat_size = 32 * img_h * img_w
            self.fc = nn.Linear(latent_dim, self.flat_size)
            self.deconv = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = self.fc(z)
            x = x.view(x.size(0), 32, img_h, img_w)
            return self.deconv(x)

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= n_samples
        losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d, Loss: %.6f", epoch + 1, epochs, epoch_loss)

    # Plot training loss
    if out_dir:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Autoencoder Training Loss")
        fig.savefig(out_dir / "autoencoder_loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Extract latent vectors
    model.eval()
    with torch.no_grad():
        all_z = model.encoder(X.to(device))
    latent_vectors = all_z.cpu().numpy()

    logger.info("Latent space shape: %s", latent_vectors.shape)
    return latent_vectors, model


# ---------------------------------------------------------------------------
# Latent space visualisation and comparison
# ---------------------------------------------------------------------------

def visualise_latent_space(
    latent_vectors: np.ndarray,
    accession_ids: list[str],
    phenotype_df: pd.DataFrame | None,
    tabular_umap_path: Path | None,
    out_dir: Path,
) -> pd.DataFrame:
    """UMAP the autoencoder latent space and compare with tabular clustering.

    Args:
        latent_vectors: Latent representations from the autoencoder.
        accession_ids: Accession IDs matching the latent vectors.
        phenotype_df: Optional phenotype data for colouring.
        tabular_umap_path: Path to the Phase 3 UMAP clusters CSV.
        out_dir: Directory to save plots.

    Returns:
        DataFrame with image-based UMAP coordinates.
    """
    # UMAP on latent vectors
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_vectors)

    img_umap = pd.DataFrame(
        embedding,
        columns=["UMAP1_img", "UMAP2_img"],
        index=accession_ids,
    )

    # Plot: image-based UMAP coloured by phenotype
    if phenotype_df is not None:
        for trait in phenotype_df.columns:
            vals = phenotype_df[trait].reindex(accession_ids)
            mask = vals.notna()
            if mask.sum() < 10:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                img_umap.loc[mask.values, "UMAP1_img"],
                img_umap.loc[mask.values, "UMAP2_img"],
                c=vals[mask].values,
                cmap="viridis",
                s=15,
                alpha=0.7,
            )
            ax.set_xlabel("UMAP 1 (image latent)")
            ax.set_ylabel("UMAP 2 (image latent)")
            ax.set_title(f"Image-based latent space: {trait}")
            plt.colorbar(scatter, ax=ax, label=trait)
            fig.savefig(out_dir / f"latent_umap_{trait}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Compare with tabular UMAP if available
    if tabular_umap_path and tabular_umap_path.exists():
        tab_umap = pd.read_csv(tabular_umap_path, index_col=0)
        common = img_umap.index.intersection(tab_umap.index)
        if len(common) > 10:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].scatter(
                tab_umap.loc[common, "UMAP1"],
                tab_umap.loc[common, "UMAP2"],
                s=10, alpha=0.5,
            )
            axes[0].set_title("Tabular IR Fingerprint (Phase 3)")
            axes[0].set_xlabel("UMAP 1")
            axes[0].set_ylabel("UMAP 2")

            axes[1].scatter(
                img_umap.loc[common, "UMAP1_img"],
                img_umap.loc[common, "UMAP2_img"],
                s=10, alpha=0.5,
            )
            axes[1].set_title("Image Autoencoder Latent (Phase 4)")
            axes[1].set_xlabel("UMAP 1")
            axes[1].set_ylabel("UMAP 2")

            fig.suptitle("Tabular vs Image-based Clustering Comparison")
            fig.savefig(out_dir / "comparison_tab_vs_img.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved tabular vs image comparison plot")

            # Quantitative comparison: Procrustes or Mantel-like correlation
            from scipy.spatial.distance import pdist
            from scipy.stats import spearmanr

            d_tab = pdist(tab_umap.loc[common, ["UMAP1", "UMAP2"]].values)
            d_img = pdist(img_umap.loc[common, ["UMAP1_img", "UMAP2_img"]].values)
            corr, pval = spearmanr(d_tab, d_img)
            logger.info(
                "Pairwise distance correlation (Spearman) between tabular and image UMAP: "
                "r=%.4f, p=%.2e",
                corr, pval,
            )

    img_umap.to_csv(out_dir / "latent_umap.csv")
    return img_umap


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase4(cfg: Config) -> dict:
    """Execute Phase 4: image encoding, autoencoder training, and comparison.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict with keys 'latent_vectors', 'img_umap'.
    """
    out_dir = resolve_path(cfg, cfg.paths.phase4_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data from previous phases
    fp_path = resolve_path(cfg, cfg.paths.phase1_output) / "combined_fingerprint.parquet"
    merged_path = resolve_path(cfg, cfg.paths.phase2_output) / "merged_data.parquet"

    if not fp_path.exists():
        raise FileNotFoundError(f"Phase 1 output not found: {fp_path}")

    fingerprint_df = pd.read_parquet(fp_path)

    phenotype_df = None
    if merged_path.exists():
        merged = pd.read_parquet(merged_path)
        trait_cols = [c for c in cfg.params.target_traits if c in merged.columns]
        if trait_cols:
            phenotype_df = merged[trait_cols]

    # Get chromosome length (approximate from IR positions)
    chrom = cfg.params.chromosomes[0]
    ir_bed_path = resolve_path(cfg, cfg.paths.phase1_output) / f"{chrom}_ir_regions.bed"
    if not ir_bed_path.exists():
        raise FileNotFoundError(f"IR BED file not found: {ir_bed_path}")

    from genophenoir.ir_profiler import load_ir_bed
    ir_list = load_ir_bed(ir_bed_path)
    chrom_length = max(ir.end for ir in ir_list) if ir_list else 30_000_000  # fallback

    # Create images
    logger.info("=== Creating accession images ===")
    images, accession_ids = create_accession_images(
        ir_bed_path=ir_bed_path,
        fingerprint_df=fingerprint_df,
        phenotype_df=phenotype_df,
        chrom=chrom,
        chrom_length=chrom_length,
        window_size=cfg.params.window_size_bp,
        out_dir=out_dir,
    )

    # Train autoencoder
    logger.info("=== Training CNN Autoencoder ===")
    latent_vectors, model = build_and_train_autoencoder(
        images,
        latent_dim=cfg.params.latent_dim,
        epochs=cfg.params.autoencoder_epochs,
        batch_size=cfg.params.autoencoder_batch_size,
        lr=cfg.params.autoencoder_lr,
        out_dir=out_dir,
    )

    # Save latent vectors
    latent_df = pd.DataFrame(
        latent_vectors,
        index=accession_ids,
        columns=[f"z_{i}" for i in range(latent_vectors.shape[1])],
    )
    latent_df.to_csv(out_dir / "latent_vectors.csv")

    # Visualise and compare
    logger.info("=== Visualising Latent Space ===")
    tab_umap_path = resolve_path(cfg, cfg.paths.phase3_output) / "umap_clusters.csv"
    img_umap = visualise_latent_space(
        latent_vectors, accession_ids, phenotype_df, tab_umap_path, out_dir,
    )

    logger.info("Phase 4 complete. Outputs saved to %s", out_dir)

    return {
        "latent_vectors": latent_vectors,
        "img_umap": img_umap,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config()
    run_phase4(cfg)
