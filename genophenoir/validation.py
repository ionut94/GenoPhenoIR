"""Phase 3b — Validation: matched-null + kinship correction.

This module answers the question that decides whether the whole IR idea has
legs: **is there phenotypic signal that is specific to inverted repeats, over
and above (a) what any matched set of genomic regions would give, and (b) what
population structure alone explains?**

Three tests, all under leave-population-out cross-validation so a model cannot
cheat by exploiting within-population relatedness:

  1. Matched-null     — predict each trait from the real IR fingerprint, then
                        from N sets of random regions with the SAME count and
                        length distribution. Empirical p = fraction of null sets
                        whose CV R^2 >= the IR CV R^2.
  2. Structure baseline — predict each trait from kinship-derived structure
                        axes alone (PCoA of the IBS kinship matrix).
  3. Added value      — compare `structure` vs `structure + IR`. If IR adds
                        nothing beyond structure, the idea is confounded.

Population labels (the CV groups) come from clustering the kinship eigenvectors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score

from genophenoir.config_loader import Config, load_config, resolve_path
from genophenoir.ir_profiler import (
    InvertedRepeat,
    load_hdf5_variants,
    load_ir_bed,
    load_vcf_variants,
)
from genophenoir.phenotype_loader import load_phenotype_data

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Kinship / population structure
# ---------------------------------------------------------------------------

def load_kinship(h5_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load the 1001G IBS kinship matrix and its accession order.

    Returns:
        Tuple of (kinship [n x n] float matrix, accession_ids).
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        K = f["kinship"][:]
        acc = [a.decode() if isinstance(a, bytes) else str(a) for a in f["accessions"][:]]
    logger.info("Loaded kinship matrix: %d x %d", *K.shape)
    return K, acc


def structure_components(K: np.ndarray, n_components: int) -> np.ndarray:
    """Principal coordinates of the kinship matrix (classical MDS / PCoA).

    The leading eigenvectors of the (double-centred) kinship matrix are the
    standard axes of population structure for *A. thaliana*.

    Args:
        K: Symmetric kinship matrix (n x n).
        n_components: Number of leading axes to return.

    Returns:
        Array of shape (n, n_components) — structure coordinates per accession.
    """
    n = K.shape[0]
    # Double-centre, then eigendecompose (symmetric -> eigh)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ K @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    k = min(n_components, n)
    pos = np.clip(eigvals[:k], 0, None)
    coords = eigvecs[:, :k] * np.sqrt(pos)
    total = np.clip(eigvals, 0, None).sum()
    if total > 0:
        logger.info(
            "Structure PCoA: top %d axes explain %.1f%% of kinship variance",
            k, 100 * pos.sum() / total,
        )
    else:
        logger.warning("Structure PCoA: kinship has no positive eigenvalues; coords degenerate")
    return coords


def assign_populations(coords: np.ndarray, n_populations: int, seed: int = 42) -> np.ndarray:
    """Cluster accessions into populations via KMeans on structure coordinates.

    Used as the groups for leave-population-out cross-validation.
    """
    k = min(n_populations, coords.shape[0])
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(coords)
    sizes = np.bincount(labels)
    logger.info("Assigned %d populations (sizes: %s)", k, sizes.tolist())
    return labels


# ---------------------------------------------------------------------------
# Region fingerprints (efficient, searchsorted-based)
# ---------------------------------------------------------------------------

def build_region_fingerprint(
    regions: list[tuple[int, int]],
    positions: np.ndarray,
    genotypes: np.ndarray,
    disruption_threshold: int = 1,
) -> np.ndarray:
    """Build a binary accession x region fingerprint efficiently.

    For each region, count alt alleles per accession among SNPs inside it; mark
    the region disrupted (0) if the count >= threshold, else intact (1).

    Assumes ``positions`` is sorted ascending (true within a chromosome block
    of the 1001G matrix) so each region is a contiguous slice found by
    binary search.

    Args:
        regions: List of (start, end) 0-based half-open intervals.
        positions: Sorted 1D array of SNP positions (length = n_variants).
        genotypes: Binary matrix (n_variants x n_accessions), 0=ref, 1=alt.
        disruption_threshold: Min alt alleles in a region to call it disrupted.

    Returns:
        Binary matrix (n_accessions x n_regions): 1 = intact, 0 = disrupted.
    """
    n_acc = genotypes.shape[1]
    fp = np.ones((n_acc, len(regions)), dtype=np.int8)
    los = np.searchsorted(positions, [r[0] for r in regions], side="left")
    his = np.searchsorted(positions, [r[1] for r in regions], side="left")
    for j, (lo, hi) in enumerate(zip(los, his)):
        if hi > lo:
            counts = genotypes[lo:hi, :].sum(axis=0)
            fp[counts >= disruption_threshold, j] = 0
    return fp


def generate_matched_regions(
    ir_list: list[InvertedRepeat],
    chrom_length: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Generate random regions matching the IR count and length distribution.

    Each real IR contributes one random region of identical length placed
    uniformly at random on the chromosome. This preserves both the number of
    features and the length distribution — the two things that most strongly
    drive how often a region happens to contain a SNP.
    """
    regions: list[tuple[int, int]] = []
    for ir in ir_list:
        length = ir.end - ir.start
        if length <= 0 or length >= chrom_length:
            continue
        start = int(rng.integers(0, chrom_length - length))
        regions.append((start, start + length))
    return regions


# ---------------------------------------------------------------------------
# Prediction under leave-population-out CV
# ---------------------------------------------------------------------------

def cv_r2(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    n_estimators: int,
    seed: int = 42,
) -> float:
    """Leave-population-out cross-validated R^2 for a RandomForest regressor.

    GroupKFold places whole populations into held-out folds, so prediction is
    always for accessions from populations unseen in training.
    """
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2 or X.shape[1] == 0:
        return float("nan")
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    )
    preds = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=splits))
    return r2_score(y, preds)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_genotypes(cfg: Config, chrom: str) -> tuple[pd.DataFrame, list[str]]:
    """Load variants for one chromosome, preferring the HDF5 matrix."""
    h5_path = resolve_path(cfg, cfg.paths.snp_matrix_hdf5) if cfg.paths.snp_matrix_hdf5 else None
    if h5_path and h5_path.exists():
        return load_hdf5_variants(h5_path, chrom, max_accessions=cfg.params.max_accessions)
    vcf_path = resolve_path(cfg, cfg.paths.vcf_file) if cfg.paths.vcf_file else None
    if vcf_path and vcf_path.exists():
        return load_vcf_variants(vcf_path, chrom, max_accessions=cfg.params.max_accessions)
    raise FileNotFoundError("No genotype source (HDF5 or VCF) found for validation")


def run_validation(cfg: Config) -> pd.DataFrame:
    """Execute Phase 3b: matched-null + kinship-corrected validation.

    Returns:
        Summary DataFrame (one row per trait) of CV R^2 for each feature set,
        the empirical p-value vs the matched null, and the structure baseline.
    """
    out_dir = resolve_path(cfg, cfg.paths.validation_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom = cfg.params.chromosomes[0]

    # --- IR coordinates (from Phase 1) ---
    bed_path = resolve_path(cfg, cfg.paths.phase1_output) / f"{chrom}_ir_regions.bed"
    if not bed_path.exists():
        raise FileNotFoundError(f"IR BED not found: {bed_path}. Run Phase 1 first.")
    ir_list = load_ir_bed(bed_path)
    logger.info("Loaded %d IRs for %s", len(ir_list), chrom)
    ir_regions = [(ir.start, ir.end) for ir in ir_list]

    # --- Genotypes ---
    variants_df, geno_acc = _load_genotypes(cfg, chrom)
    positions = variants_df["pos"].to_numpy()
    genotypes_full = variants_df[geno_acc].to_numpy(dtype=np.int8)  # (n_var x n_acc)
    chrom_length = int(positions.max()) + 1
    logger.info("Genotypes: %d variants x %d accessions", *genotypes_full.shape)

    # --- Phenotypes ---
    pheno_path = resolve_path(cfg, cfg.paths.phenotype_csv)
    pheno = load_phenotype_data(pheno_path, cfg.params.target_traits)
    pheno.index = pheno.index.astype(str)

    # --- Kinship / structure ---
    kin_path = resolve_path(cfg, cfg.paths.kinship_hdf5)
    K_full, kin_acc = load_kinship(kin_path)

    # --- Align all sources to a common, ordered accession set ---
    geno_pos = {a: i for i, a in enumerate(geno_acc)}
    kin_pos = {a: i for i, a in enumerate(kin_acc)}
    common = [a for a in geno_acc if a in kin_pos and a in pheno.index]
    if len(common) < 50:
        raise RuntimeError(f"Only {len(common)} accessions shared across all sources")
    logger.info("Common accessions across genotype/kinship/phenotype: %d", len(common))

    g_idx = [geno_pos[a] for a in common]
    k_idx = [kin_pos[a] for a in common]
    genotypes = genotypes_full[:, g_idx]
    K = K_full[np.ix_(k_idx, k_idx)]
    pheno = pheno.loc[common]

    # --- Population structure ---
    coords = structure_components(K, cfg.params.n_structure_pcs)
    groups = assign_populations(coords, cfg.params.n_populations)

    # --- Real IR fingerprint ---
    ir_fp = build_region_fingerprint(
        ir_regions, positions, genotypes, cfg.params.ir_disruption_threshold
    ).astype(np.float32)
    # Drop zero-variance columns (uninformative for any model)
    ir_fp = ir_fp[:, ir_fp.std(axis=0) > 0]
    logger.info("IR fingerprint (variable cols): %d x %d", *ir_fp.shape)

    rng = np.random.default_rng(42)
    rows = []
    for trait in pheno.columns:
        y_full = pheno[trait]
        mask = y_full.notna().to_numpy()
        if mask.sum() < 20:
            logger.warning("Skipping %s: only %d samples", trait, int(mask.sum()))
            continue
        y = y_full.to_numpy()[mask]
        g = groups[mask]
        splits = cfg.params.cv_folds
        n_est = cfg.params.validation_n_estimators

        logger.info("=== %s (n=%d) ===", trait, len(y))

        # 1. IR fingerprint
        r2_ir = cv_r2(ir_fp[mask], y, g, splits, n_est)

        # 2. Structure baseline
        r2_struct = cv_r2(coords[mask], y, g, splits, n_est)

        # 3. Structure + IR
        r2_struct_ir = cv_r2(
            np.hstack([coords[mask], ir_fp[mask]]), y, g, splits, n_est
        )

        # 4. Matched-null distribution
        null_r2 = []
        for n in range(cfg.params.n_null_regions):
            rand_regions = generate_matched_regions(ir_list, chrom_length, rng)
            rand_fp = build_region_fingerprint(
                rand_regions, positions, genotypes, cfg.params.ir_disruption_threshold
            ).astype(np.float32)
            rand_fp = rand_fp[:, rand_fp.std(axis=0) > 0]
            null_r2.append(cv_r2(rand_fp[mask], y, g, splits, n_est))
        null_r2 = np.array(null_r2, dtype=float)
        null_valid = null_r2[~np.isnan(null_r2)]

        # Empirical p-value: how often does a random region set match/beat IR?
        if len(null_valid) and not np.isnan(r2_ir):
            p_emp = (1 + np.sum(null_valid >= r2_ir)) / (1 + len(null_valid))
        else:
            p_emp = float("nan")

        logger.info(
            "%s | IR R2=%.4f | null R2=%.4f+/-%.4f (max %.4f) | p=%.3f | "
            "struct=%.4f | struct+IR=%.4f",
            trait, r2_ir, null_valid.mean() if len(null_valid) else float("nan"),
            null_valid.std() if len(null_valid) else float("nan"),
            null_valid.max() if len(null_valid) else float("nan"),
            p_emp, r2_struct, r2_struct_ir,
        )

        rows.append({
            "trait": trait,
            "n_samples": len(y),
            "r2_ir": r2_ir,
            "r2_null_mean": null_valid.mean() if len(null_valid) else np.nan,
            "r2_null_std": null_valid.std() if len(null_valid) else np.nan,
            "r2_null_max": null_valid.max() if len(null_valid) else np.nan,
            "p_empirical": p_emp,
            "r2_structure": r2_struct,
            "r2_structure_plus_ir": r2_struct_ir,
            "ir_adds_over_structure": r2_struct_ir - r2_struct,
            "n_null_sets": len(null_valid),
        })

        # Plot: null distribution with IR + structure markers
        _plot_null(trait, null_valid, r2_ir, r2_struct, out_dir)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "validation_summary.csv", index=False)
    logger.info("Validation summary:\n%s", summary.to_string(index=False))
    _print_verdict(summary)
    return summary


def _plot_null(
    trait: str,
    null_r2: np.ndarray,
    r2_ir: float,
    r2_struct: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(null_r2):
        ax.hist(null_r2, bins=20, color="0.7", edgecolor="white", label="Matched-random regions (null)")
    ax.axvline(r2_ir, color="crimson", lw=2, label=f"IR fingerprint (R²={r2_ir:.3f})")
    ax.axvline(r2_struct, color="steelblue", lw=2, ls="--", label=f"Structure only (R²={r2_struct:.3f})")
    ax.set_xlabel("Leave-population-out CV R²")
    ax.set_ylabel("Count")
    ax.set_title(f"Matched-null validation: {trait}")
    ax.legend()
    fig.savefig(out_dir / f"null_distribution_{trait}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_verdict(summary: pd.DataFrame) -> None:
    """Emit a plain-language verdict per trait."""
    if summary.empty:
        logger.warning("No traits evaluated; cannot produce a verdict.")
        return
    logger.info("===== VERDICT =====")
    for _, r in summary.iterrows():
        ir_specific = (not np.isnan(r["p_empirical"])) and r["p_empirical"] < 0.05
        beats_struct = r["ir_adds_over_structure"] > 0.01
        if ir_specific and beats_struct:
            msg = "IR-SPECIFIC signal that adds over population structure — worth pursuing."
        elif ir_specific and not beats_struct:
            msg = "Beats matched-random but NOT structure — likely confounded by kinship."
        elif (not ir_specific) and r["r2_ir"] <= 0:
            msg = "No predictive signal at all (R² <= 0)."
        else:
            msg = "No IR-specific signal: indistinguishable from random matched regions."
        logger.info("  %s: %s", r["trait"], msg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config()
    run_validation(cfg)
