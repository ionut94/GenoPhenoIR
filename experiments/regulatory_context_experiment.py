"""Experiment 03 — Regulatory context of inverted repeats.

Two questions:

  (A) Genome knowledge: are IRs *enriched* in regulatory regions (promoters,
      UTRs) relative to random genomic placement? IRs in promoters can form
      cruciforms that modulate transcription, so positional enrichment there
      would be biologically meaningful independent of phenotype.

  (B) Phenotype link: does the disruption of IRs in a given regulatory context
      (e.g. promoter IRs only) predict flowering time better than matched-random
      regions, even though the bulk of IRs do not (Exp 01/02)?

Regulatory context per IR (by midpoint, priority order):
  promoter (<=1 kb upstream of TSS) > 5'UTR > 3'UTR > exon/CDS > intron > intergenic

Outputs:
  reports/experiment-03-results.csv          context-stratified prediction
  reports/experiment-03-enrichment.csv       IR enrichment per context
  reports/figures/exp03_enrichment.png       observed vs expected IR counts
  reports/figures/exp03_context_forest.png   real vs null R² per context

Run from repo root:  python experiments/regulatory_context_experiment.py
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
from genophenoir.ir_profiler import load_ir_bed
from genophenoir.ir_features import build_feature_fingerprints, generate_matched_irs
from genophenoir.phenotype_loader import load_phenotype_data
from genophenoir.validation import (
    _load_genotypes, assign_populations, cv_r2, load_kinship, structure_components,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp03")
sns.set_theme(style="whitegrid", context="notebook")

FEATURE = "stem_snp"
PROMOTER_BP = 1000
CONTEXTS = ["promoter", "five_prime_UTR", "three_prime_UTR", "exon", "intron", "intergenic"]


def build_context_masks(gff_path: Path, chrom: str, chrom_len: int) -> dict[str, np.ndarray]:
    """Boolean coverage masks over the chromosome for each regulatory context."""
    gff_chrom = chrom.replace("Chr", "")  # GFF uses '1'
    masks = {c: np.zeros(chrom_len, dtype=bool) for c in
             ["promoter", "five_prime_UTR", "three_prime_UTR", "exon", "gene_body"]}
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[0] != gff_chrom:
                continue
            ftype, start, end, strand = p[2], int(p[3]) - 1, int(p[4]), p[6]
            start = max(0, start); end = min(chrom_len, end)
            if ftype == "gene":
                masks["gene_body"][start:end] = True
                if strand == "+":
                    masks["promoter"][max(0, start - PROMOTER_BP):start] = True
                else:
                    masks["promoter"][end:min(chrom_len, end + PROMOTER_BP)] = True
            elif ftype == "five_prime_UTR":
                masks["five_prime_UTR"][start:end] = True
            elif ftype == "three_prime_UTR":
                masks["three_prime_UTR"][start:end] = True
            elif ftype in ("exon", "CDS"):
                masks["exon"][start:end] = True
    return masks


def assign_context(midpoint: int, masks: dict[str, np.ndarray]) -> str:
    """Assign one context to a point by biological priority."""
    if masks["promoter"][midpoint]:
        return "promoter"
    if masks["five_prime_UTR"][midpoint]:
        return "five_prime_UTR"
    if masks["three_prime_UTR"][midpoint]:
        return "three_prime_UTR"
    if masks["exon"][midpoint]:
        return "exon"
    if masks["gene_body"][midpoint]:
        return "intron"
    return "intergenic"


def main() -> None:
    cfg = load_config()
    chrom = cfg.params.chromosomes[0]
    n_null, n_est, splits = cfg.params.n_null_regions, cfg.params.validation_n_estimators, cfg.params.cv_folds
    reports = Path("reports"); figs = reports / "figures"; figs.mkdir(parents=True, exist_ok=True)

    irs = load_ir_bed(resolve_path(cfg, cfg.paths.phase1_output) / f"{chrom}_ir_regions.bed")
    variants_df, geno_acc = _load_genotypes(cfg, chrom)
    positions = variants_df["pos"].to_numpy()
    genotypes_full = variants_df[geno_acc].to_numpy(dtype=np.int8)
    chrom_length = int(positions.max()) + 1

    fasta = resolve_path(cfg, cfg.paths.reference_fasta)
    reference = next(str(r.seq).upper() for r in SeqIO.parse(fasta, "fasta")
                     if r.id.split()[0] in (chrom, chrom.replace("Chr", "")))
    chrom_len = len(reference)

    # --- Context masks + IR annotation ---
    gff = resolve_path(cfg, cfg.paths.gff3_file)
    masks = build_context_masks(gff, chrom, chrom_len)
    ir_ctx = np.array([assign_context((ir.start + ir.end) // 2, masks) for ir in irs])
    logger.info("IR context counts: %s",
                {c: int((ir_ctx == c).sum()) for c in CONTEXTS})

    # --- (A) Enrichment: observed IR counts per context vs random placement ---
    rng = np.random.default_rng(7)
    obs = {c: int((ir_ctx == c).sum()) for c in CONTEXTS}
    n_perm = 200
    perm_counts = {c: np.zeros(n_perm) for c in CONTEXTS}
    lengths = np.array([ir.end - ir.start for ir in irs])
    for i in range(n_perm):
        starts = rng.integers(0, chrom_len - lengths)
        mids = starts + lengths // 2
        ctx = np.empty(len(irs), dtype=object)
        # vectorised context lookup via the masks
        prom = masks["promoter"][mids]; f5 = masks["five_prime_UTR"][mids]
        f3 = masks["three_prime_UTR"][mids]; ex = masks["exon"][mids]; gb = masks["gene_body"][mids]
        ctx[:] = "intergenic"
        ctx[gb] = "intron"; ctx[ex] = "exon"; ctx[f3] = "three_prime_UTR"
        ctx[f5] = "five_prime_UTR"; ctx[prom] = "promoter"
        for c in CONTEXTS:
            perm_counts[c][i] = (ctx == c).sum()
    enr_rows = []
    for c in CONTEXTS:
        exp_mean = perm_counts[c].mean()
        p_enr = (1 + np.sum(perm_counts[c] >= obs[c])) / (1 + n_perm)
        enr_rows.append({"context": c, "observed": obs[c], "expected_mean": exp_mean,
                         "expected_std": perm_counts[c].std(),
                         "fold_enrichment": obs[c] / max(1e-9, exp_mean),
                         "p_enrichment": p_enr})
    enrichment = pd.DataFrame(enr_rows)
    enrichment.to_csv(reports / "experiment-03-enrichment.csv", index=False)
    logger.info("ENRICHMENT:\n%s", enrichment.to_string(index=False))

    # --- (B) Context-stratified matched-null prediction ---
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
    traits = list(pheno.columns)
    masks_t = {t: pheno[t].notna().to_numpy() for t in traits}

    rng2 = np.random.default_rng(42)
    rows = []
    for c in CONTEXTS:
        class_irs = [ir for ir, cc in zip(irs, ir_ctx) if cc == c]
        if len(class_irs) < 50:
            logger.info("Skipping context %s (only %d IRs)", c, len(class_irs))
            continue
        real_fp = build_feature_fingerprints(class_irs, positions, genotypes, reference,
                                              cfg.params.ir_disruption_threshold)[FEATURE]
        real_fp = real_fp[:, real_fp.std(axis=0) > 0]
        for t in traits:
            m = masks_t[t]; y, g = pheno[t].to_numpy()[m], groups[m]
            r2 = cv_r2(real_fp[m], y, g, splits, n_est)
            null = []
            for _ in range(n_null):
                reloc = generate_matched_irs(class_irs, chrom_length, rng2)
                nfp = build_feature_fingerprints(reloc, positions, genotypes, reference,
                                                 cfg.params.ir_disruption_threshold)[FEATURE]
                nfp = nfp[:, nfp.std(axis=0) > 0]
                null.append(cv_r2(nfp[m], y, g, splits, n_est))
            null = np.array([v for v in null if not np.isnan(v)])
            p = (1 + np.sum(null >= r2)) / (1 + len(null)) if len(null) and not np.isnan(r2) else np.nan
            rows.append({"context": c, "n_irs": len(class_irs), "trait": t, "r2_real": r2,
                         "r2_null_mean": null.mean() if len(null) else np.nan,
                         "r2_null_std": null.std() if len(null) else np.nan,
                         "p_empirical": p, "beats_null": bool(not np.isnan(p) and p < 0.05)})
            logger.info("%s | %s | n=%d | R2=%.3f null=%.3f p=%.3f", c, t, len(class_irs),
                        r2, null.mean() if len(null) else float("nan"), p)

    summary = pd.DataFrame(rows)
    summary.to_csv(reports / "experiment-03-results.csv", index=False)
    logger.info("PREDICTION:\n%s", summary.to_string(index=False))

    _fig_enrichment(enrichment, figs / "exp03_enrichment.png")
    if not summary.empty:
        _fig_forest(summary, traits, figs / "exp03_context_forest.png")
    logger.info("Figures written to %s", figs)
    print("EXPERIMENT_COMPLETE")


def _fig_enrichment(enr, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(enr))
    ax.bar(x - 0.2, enr["observed"], 0.4, label="Observed IRs", color="#c0392b")
    ax.bar(x + 0.2, enr["expected_mean"], 0.4, yerr=enr["expected_std"],
           label="Expected (random)", color="0.7", capsize=4)
    for xi, (_, r) in zip(x, enr.iterrows()):
        ax.text(xi, max(r["observed"], r["expected_mean"]) * 1.02,
                f"{r['fold_enrichment']:.2f}x\np={r['p_enrichment']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(enr["context"], rotation=20)
    ax.set_ylabel("IR count"); ax.set_title("Experiment 03A — IR enrichment by regulatory context")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _fig_forest(summary, traits, path):
    fig, axes = plt.subplots(1, len(traits), figsize=(7 * len(traits), 5), squeeze=False, sharey=True)
    for jx, t in enumerate(traits):
        ax = axes[0][jx]; sub = summary[summary["trait"] == t].iloc[::-1]
        ypos = np.arange(len(sub))
        ax.errorbar(sub["r2_null_mean"], ypos, xerr=sub["r2_null_std"], fmt="o", color="0.6", capsize=3, label="Null")
        ax.scatter(sub["r2_real"], ypos, color="crimson", zorder=3, s=55, label="Real IR context")
        for yi, (_, r) in zip(ypos, sub.iterrows()):
            star = " *" if r["beats_null"] else ""
            ax.text(max(r["r2_real"], r["r2_null_mean"]) + 0.012, yi, f"p={r['p_empirical']:.2f}{star}", va="center", fontsize=8)
        ax.set_yticks(ypos); ax.set_yticklabels(sub["context"] + " (n=" + sub["n_irs"].astype(str) + ")", fontsize=9)
        ax.set_xlabel("Leave-population-out CV R²"); ax.set_title(t); ax.legend(loc="lower right")
    fig.suptitle("Experiment 03B — IR regulatory context vs matched-random null", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
