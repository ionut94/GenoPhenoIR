"""Experiment 01 — Do mechanistic IR features carry IR-specific signal?

Compares three IR disruption features against matched random-region nulls
under leave-population-out CV, for flowering time (FT10/FT16) on Chr1:

  any_snp   — binary, any SNP in the region (the original, confounded feature)
  stem_snp  — binary, only stem-arm SNPs (loop/spacer ignored)
  ddg       — continuous ΔΔG-like stem-destabilisation score

Outputs:
  reports/experiment-01-results.csv          machine-readable results
  reports/figures/exp01_null_grid.png        null distributions (feature x trait)
  reports/figures/exp01_feature_bars.png     real vs null R² per feature
  reports/figures/exp01_ddg_diagnostics.png  feature sanity / ΔΔG illustration

Run from the repo root:  python experiments/mechanistic_experiment.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO

from genophenoir.config_loader import load_config, resolve_path
from genophenoir.ir_profiler import load_ir_bed
from genophenoir.ir_features import build_feature_fingerprints, generate_matched_irs
from genophenoir.phenotype_loader import load_phenotype_data
from genophenoir.validation import (
    _load_genotypes,
    assign_populations,
    cv_r2,
    load_kinship,
    structure_components,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp01")
sns.set_theme(style="whitegrid", context="notebook")

FEATURES = ["any_snp", "stem_snp", "ddg"]
FEATURE_LABELS = {
    "any_snp": "Any-SNP (baseline)",
    "stem_snp": "Stem-only SNP",
    "ddg": "ΔΔG-weighted",
}


def _variable_cols(mat: np.ndarray) -> np.ndarray:
    """Keep only columns with non-zero variance (informative for a model)."""
    return mat[:, mat.std(axis=0) > 0]


def main() -> None:
    cfg = load_config()
    chrom = cfg.params.chromosomes[0]
    n_null = cfg.params.n_null_regions
    n_est = cfg.params.validation_n_estimators
    splits = cfg.params.cv_folds

    reports = Path("reports")
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # --- Load IRs, genotypes, reference, phenotypes, kinship ---
    bed = resolve_path(cfg, cfg.paths.phase1_output) / f"{chrom}_ir_regions.bed"
    irs = load_ir_bed(bed)
    logger.info("Loaded %d IRs for %s", len(irs), chrom)

    variants_df, geno_acc = _load_genotypes(cfg, chrom)
    positions = variants_df["pos"].to_numpy()
    genotypes_full = variants_df[geno_acc].to_numpy(dtype=np.int8)
    chrom_length = int(positions.max()) + 1

    fasta = resolve_path(cfg, cfg.paths.reference_fasta)
    reference = None
    for rec in SeqIO.parse(fasta, "fasta"):
        name = rec.id.split()[0]
        if name == chrom or name == chrom.replace("Chr", "") or f"Chr{name}" == chrom:
            reference = str(rec.seq).upper()
            break
    logger.info("Reference %s length %d", chrom, len(reference))

    pheno = load_phenotype_data(resolve_path(cfg, cfg.paths.phenotype_csv), cfg.params.target_traits)
    pheno.index = pheno.index.astype(str)

    K_full, kin_acc = load_kinship(resolve_path(cfg, cfg.paths.kinship_hdf5))

    # --- Align accessions ---
    geno_pos = {a: i for i, a in enumerate(geno_acc)}
    kin_pos = {a: i for i, a in enumerate(kin_acc)}
    common = [a for a in geno_acc if a in kin_pos and a in pheno.index]
    logger.info("Common accessions: %d", len(common))
    genotypes = genotypes_full[:, [geno_pos[a] for a in common]]
    K = K_full[np.ix_([kin_pos[a] for a in common], [kin_pos[a] for a in common])]
    pheno = pheno.loc[common]

    coords = structure_components(K, cfg.params.n_structure_pcs)
    groups = assign_populations(coords, cfg.params.n_populations)

    # --- Real feature fingerprints ---
    logger.info("Building real feature fingerprints...")
    real = build_feature_fingerprints(irs, positions, genotypes, reference,
                                      cfg.params.ir_disruption_threshold)

    # --- Diagnostics: stem vs loop SNP classification, ΔΔG distribution ---
    diag = _diagnostics(irs, positions, genotypes, reference)

    # --- Per-trait: real R² + structure baseline + matched-null distribution ---
    rng = np.random.default_rng(42)
    rows = []
    null_store: dict[tuple[str, str], np.ndarray] = {}
    real_store: dict[tuple[str, str], float] = {}

    traits = list(pheno.columns)
    masks = {t: pheno[t].notna().to_numpy() for t in traits}

    # Real R² per feature/trait + structure baseline
    struct_r2 = {}
    for t in traits:
        m = masks[t]
        y, g = pheno[t].to_numpy()[m], groups[m]
        struct_r2[t] = cv_r2(coords[m], y, g, splits, n_est)
        for feat in FEATURES:
            r2 = cv_r2(_variable_cols(real[feat][m]), y, g, splits, n_est)
            real_store[(feat, t)] = r2
            logger.info("REAL %s | %s R2=%.4f (struct=%.4f)", feat, t, r2, struct_r2[t])

    # Matched-null sweep: one relocation -> all features -> all traits
    null_lists: dict[tuple[str, str], list[float]] = {(f, t): [] for f in FEATURES for t in traits}
    for n in range(n_null):
        relocated = generate_matched_irs(irs, chrom_length, rng)
        nfp = build_feature_fingerprints(relocated, positions, genotypes, reference,
                                         cfg.params.ir_disruption_threshold)
        for t in traits:
            m = masks[t]
            y, g = pheno[t].to_numpy()[m], groups[m]
            for feat in FEATURES:
                null_lists[(feat, t)].append(cv_r2(_variable_cols(nfp[feat][m]), y, g, splits, n_est))
        logger.info("null %d/%d done", n + 1, n_null)

    for feat in FEATURES:
        for t in traits:
            null = np.array([v for v in null_lists[(feat, t)] if not np.isnan(v)])
            null_store[(feat, t)] = null
            r2 = real_store[(feat, t)]
            p = (1 + np.sum(null >= r2)) / (1 + len(null)) if len(null) and not np.isnan(r2) else np.nan
            rows.append({
                "feature": feat, "trait": t, "r2_real": r2,
                "r2_null_mean": null.mean() if len(null) else np.nan,
                "r2_null_std": null.std() if len(null) else np.nan,
                "r2_null_max": null.max() if len(null) else np.nan,
                "p_empirical": p, "r2_structure": struct_r2[t],
                "beats_null": bool(not np.isnan(p) and p < 0.05),
                "n_null": len(null),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(reports / "experiment-01-results.csv", index=False)
    (reports / "experiment-01-diagnostics.json").write_text(json.dumps(diag, indent=2))
    logger.info("RESULTS:\n%s", summary.to_string(index=False))

    # --- Figures ---
    _fig_null_grid(null_store, real_store, struct_r2, traits, figs / "exp01_null_grid.png")
    _fig_feature_bars(summary, figs / "exp01_feature_bars.png")
    _fig_ddg_diagnostics(diag, real, figs / "exp01_ddg_diagnostics.png")
    logger.info("Figures written to %s", figs)
    print("EXPERIMENT_COMPLETE")


def _diagnostics(irs, positions, genotypes, reference) -> dict:
    """Count stem vs loop SNPs and summarise the ΔΔG feature."""
    los = np.searchsorted(positions, [ir.start for ir in irs], "left")
    his = np.searchsorted(positions, [ir.end for ir in irs], "left")
    stem_snps = loop_snps = 0
    for j, ir in enumerate(irs):
        lo, hi = int(los[j]), int(his[j])
        S = ir.stem_length
        for p in positions[lo:hi]:
            p = int(p)
            if ir.start <= p < ir.start + S or ir.end - S <= p < ir.end:
                stem_snps += 1
            else:
                loop_snps += 1
    return {
        "n_irs": len(irs),
        "stem_snp_sites": int(stem_snps),
        "loop_spacer_snp_sites": int(loop_snps),
        "stem_fraction": round(stem_snps / max(1, stem_snps + loop_snps), 3),
    }


def _fig_null_grid(null_store, real_store, struct_r2, traits, path):
    fig, axes = plt.subplots(len(FEATURES), len(traits), figsize=(11, 10), squeeze=False)
    for i, feat in enumerate(FEATURES):
        for jx, t in enumerate(traits):
            ax = axes[i][jx]
            null = null_store[(feat, t)]
            r2 = real_store[(feat, t)]
            if len(null):
                ax.hist(null, bins=15, color="0.75", edgecolor="white")
            ax.axvline(r2, color="crimson", lw=2)
            ax.axvline(struct_r2[t], color="steelblue", lw=1.5, ls="--")
            p = (1 + np.sum(null >= r2)) / (1 + len(null)) if len(null) else float("nan")
            ax.set_title(f"{FEATURE_LABELS[feat]} — {t}\nreal R²={r2:.3f}, p={p:.3f}", fontsize=10)
            if i == len(FEATURES) - 1:
                ax.set_xlabel("Leave-population-out CV R²")
    fig.suptitle("Real IR features (red) vs matched-random null (grey); structure-only = blue dashed",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_feature_bars(summary, path):
    traits = sorted(summary["trait"].unique())
    fig, axes = plt.subplots(1, len(traits), figsize=(6 * len(traits), 5), squeeze=False)
    x = np.arange(len(FEATURES))
    for jx, t in enumerate(traits):
        ax = axes[0][jx]
        sub = summary[summary["trait"] == t].set_index("feature").loc[FEATURES]
        ax.bar(x - 0.2, sub["r2_real"], width=0.4, label="Real IRs", color="crimson")
        ax.bar(x + 0.2, sub["r2_null_mean"], width=0.4, yerr=sub["r2_null_std"],
               label="Matched-random null", color="0.7", capsize=4)
        for xi, (_, r) in zip(x, sub.iterrows()):
            ax.text(xi, max(r["r2_real"], r["r2_null_mean"]) + 0.005, f"p={r['p_empirical']:.2f}",
                    ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES], rotation=15)
        ax.set_ylabel("CV R²")
        ax.set_title(t)
        ax.legend()
    fig.suptitle("Do IR features beat matched-random regions?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_ddg_diagnostics(diag, real, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # left: stem vs loop SNP sites
    axes[0].bar(["Stem-arm", "Loop/spacer"],
                [diag["stem_snp_sites"], diag["loop_spacer_snp_sites"]],
                color=["#c44", "#aaa"])
    axes[0].set_ylabel("SNP sites across all IRs")
    axes[0].set_title(f"SNP location within IRs\n(stem fraction = {diag['stem_fraction']})")
    # right: ΔΔG score distribution (non-zero entries)
    vals = real["ddg"].ravel()
    vals = vals[vals > 0]
    axes[1].hist(vals, bins=40, color="#369")
    axes[1].set_xlabel("Per-accession ΔΔG destabilisation score (non-zero)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"ΔΔG feature distribution\n(mean non-zero = {vals.mean():.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
