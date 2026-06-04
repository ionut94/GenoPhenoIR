"""Experiment 02 — Do specific structural classes of IR carry signal?

Experiment 01 lumped all 6642 IRs together and found no IR-specific signal.
Maybe a *functional subset* does. IUPACpal gives us the geometry to slice IRs
into structural classes — by stem length, loop size, mismatch count (degeneracy),
and stem GC — and test each class against its own matched-random null under
leave-population-out CV.

Hypothesis: a structurally distinct class (e.g. long, perfect, tight-loop
hairpins — the strongest cruciform formers) carries flowering-time signal even
though the bulk of IRs do not.

Outputs:
  reports/experiment-02-results.csv
  reports/figures/exp02_class_forest.png    real vs null R² per class (forest plot)
  reports/figures/exp02_class_sizes.png     class sizes + geometry

Run from repo root:  python experiments/structural_classes_experiment.py
"""

from __future__ import annotations

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
from genophenoir.ir_profiler import load_ir_bed, reverse_complement
from genophenoir.ir_features import build_feature_fingerprints, generate_matched_irs
from genophenoir.phenotype_loader import load_phenotype_data
from genophenoir.validation import (
    _load_genotypes, assign_populations, cv_r2, load_kinship, structure_components,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp02")
sns.set_theme(style="whitegrid", context="notebook")

FEATURE = "stem_snp"  # mechanistic-but-simple per-IR feature used throughout


def stem_mismatches(ir, reference: str) -> int:
    """Mismatches between the two stem arms, computed from the reference."""
    S = ir.stem_length
    left = reference[ir.start:ir.start + S]
    right = reference[ir.end - S:ir.end]
    rc = reverse_complement(left)
    return sum(1 for a, b in zip(rc, right) if a != b)


def stem_gc(ir, reference: str) -> float:
    seg = reference[ir.start:ir.start + ir.stem_length]
    return (seg.count("G") + seg.count("C")) / max(1, len(seg))


def classify(irs, reference):
    """Annotate each IR with mismatches, GC, and return class membership lists."""
    mm = np.array([stem_mismatches(ir, reference) for ir in irs])
    gc = np.array([stem_gc(ir, reference) for ir in irs])
    stem = np.array([ir.stem_length for ir in irs])
    loop = np.array([ir.spacer_length for ir in irs])

    classes = {
        "all": np.ones(len(irs), bool),
        "stem_short(14-19)": (stem >= 14) & (stem <= 19),
        "stem_medium(20-29)": (stem >= 20) & (stem <= 29),
        "stem_long(>=30)": stem >= 30,
        "loop_tight(<=3)": loop <= 3,
        "loop_large(>=16)": loop >= 16,
        "perfect(0mm)": mm == 0,
        "imperfect(>=1mm)": mm >= 1,
        "GC_rich_stem(>=0.6)": gc >= 0.6,
        "strong_hairpin(stem>=25,loop<=10,perfect)": (stem >= 25) & (loop <= 10) & (mm == 0),
    }
    return classes, {"mismatches": mm, "gc": gc}


def main() -> None:
    cfg = load_config()
    chrom = cfg.params.chromosomes[0]
    n_null, n_est, splits = cfg.params.n_null_regions, cfg.params.validation_n_estimators, cfg.params.cv_folds
    reports = Path("reports"); figs = reports / "figures"; figs.mkdir(parents=True, exist_ok=True)

    # --- Load everything (mirrors validation/exp01) ---
    irs = load_ir_bed(resolve_path(cfg, cfg.paths.phase1_output) / f"{chrom}_ir_regions.bed")
    variants_df, geno_acc = _load_genotypes(cfg, chrom)
    positions = variants_df["pos"].to_numpy()
    genotypes_full = variants_df[geno_acc].to_numpy(dtype=np.int8)
    chrom_length = int(positions.max()) + 1

    fasta = resolve_path(cfg, cfg.paths.reference_fasta)
    reference = next(
        str(r.seq).upper() for r in SeqIO.parse(fasta, "fasta")
        if r.id.split()[0] in (chrom, chrom.replace("Chr", ""))
    )

    pheno = load_phenotype_data(resolve_path(cfg, cfg.paths.phenotype_csv), cfg.params.target_traits)
    pheno.index = pheno.index.astype(str)
    K_full, kin_acc = load_kinship(resolve_path(cfg, cfg.paths.kinship_hdf5))

    geno_pos = {a: i for i, a in enumerate(geno_acc)}
    kin_pos = {a: i for i, a in enumerate(kin_acc)}
    common = [a for a in geno_acc if a in kin_pos and a in pheno.index]
    genotypes = genotypes_full[:, [geno_pos[a] for a in common]]
    K = K_full[np.ix_([kin_pos[a] for a in common], [kin_pos[a] for a in common])]
    pheno = pheno.loc[common]
    coords = structure_components(K, cfg.params.n_structure_pcs)
    groups = assign_populations(coords, cfg.params.n_populations)
    logger.info("Common accessions: %d", len(common))

    classes, _ = classify(irs, reference)
    traits = list(pheno.columns)
    masks = {t: pheno[t].notna().to_numpy() for t in traits}

    rng = np.random.default_rng(42)
    rows = []
    for cname, mask_arr in classes.items():
        class_irs = [ir for ir, keep in zip(irs, mask_arr) if keep]
        if len(class_irs) < 50:
            logger.info("Skipping class %s (only %d IRs)", cname, len(class_irs))
            continue
        # real feature matrix for this class
        real_fp = build_feature_fingerprints(class_irs, positions, genotypes, reference,
                                              cfg.params.ir_disruption_threshold)[FEATURE]
        real_fp = real_fp[:, real_fp.std(axis=0) > 0]

        for t in traits:
            m = masks[t]; y, g = pheno[t].to_numpy()[m], groups[m]
            r2_real = cv_r2(real_fp[m], y, g, splits, n_est)
            null = []
            for _ in range(n_null):
                reloc = generate_matched_irs(class_irs, chrom_length, rng)
                nfp = build_feature_fingerprints(reloc, positions, genotypes, reference,
                                                 cfg.params.ir_disruption_threshold)[FEATURE]
                nfp = nfp[:, nfp.std(axis=0) > 0]
                null.append(cv_r2(nfp[m], y, g, splits, n_est))
            null = np.array([v for v in null if not np.isnan(v)])
            p = (1 + np.sum(null >= r2_real)) / (1 + len(null)) if len(null) and not np.isnan(r2_real) else np.nan
            rows.append({
                "ir_class": cname, "n_irs": len(class_irs), "trait": t,
                "r2_real": r2_real, "r2_null_mean": null.mean() if len(null) else np.nan,
                "r2_null_std": null.std() if len(null) else np.nan,
                "p_empirical": p, "beats_null": bool(not np.isnan(p) and p < 0.05),
            })
            logger.info("%s | %s | n=%d | R2=%.3f null=%.3f p=%.3f",
                        cname, t, len(class_irs), r2_real,
                        null.mean() if len(null) else float("nan"), p)

    summary = pd.DataFrame(rows)
    summary.to_csv(reports / "experiment-02-results.csv", index=False)
    logger.info("RESULTS:\n%s", summary.to_string(index=False))

    _fig_forest(summary, traits, figs / "exp02_class_forest.png")
    _fig_sizes(classes, irs, figs / "exp02_class_sizes.png")
    logger.info("Figures written to %s", figs)
    print("EXPERIMENT_COMPLETE")


def _fig_forest(summary, traits, path):
    fig, axes = plt.subplots(1, len(traits), figsize=(7 * len(traits), 6), squeeze=False, sharey=True)
    for jx, t in enumerate(traits):
        ax = axes[0][jx]
        sub = summary[summary["trait"] == t].iloc[::-1]
        ypos = np.arange(len(sub))
        ax.errorbar(sub["r2_null_mean"], ypos, xerr=sub["r2_null_std"], fmt="o",
                    color="0.6", capsize=3, label="Matched-random null")
        ax.scatter(sub["r2_real"], ypos, color="crimson", zorder=3, s=55, label="Real IR class")
        for yi, (_, r) in zip(ypos, sub.iterrows()):
            star = " *" if r["beats_null"] else ""
            ax.text(max(r["r2_real"], r["r2_null_mean"]) + 0.012, yi,
                    f"p={r['p_empirical']:.2f}{star}", va="center", fontsize=8)
        ax.set_yticks(ypos); ax.set_yticklabels(sub["ir_class"] + " (n=" + sub["n_irs"].astype(str) + ")", fontsize=8)
        ax.set_xlabel("Leave-population-out CV R²"); ax.set_title(t); ax.legend(loc="lower right")
    fig.suptitle("Experiment 02 — IR structural classes vs matched-random null", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _fig_sizes(classes, irs, path):
    names = [c for c in classes if c != "all"]
    sizes = [int(classes[c].sum()) for c in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names[::-1], sizes[::-1], color="#5b8db8")
    ax.set_xlabel("Number of IRs"); ax.set_title(f"IR structural-class sizes (of {len(irs)} total)")
    for i, v in enumerate(sizes[::-1]):
        ax.text(v + 20, i, str(v), va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
