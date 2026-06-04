"""Experiment 04 — Are inverted-repeat stems under selective constraint?

Exp 03 showed IRs are non-randomly placed (depleted in coding, enriched in
promoters/intergenic). This asks a population-genetic question about the IRs
themselves, computable from allele frequencies alone (no new data, no alt bases
needed): are IR **stem arms** more conserved (less polymorphic) than expected —
the signature of selection maintaining the hairpin structure?

Three tests, all using per-SNP alt-allele frequency from the 1001G matrix:

  (1) Within-IR control: stem-arm polymorphism rate vs the same IR's loop/spacer.
      The loop controls for local mutation rate, so stem < loop ⇒ pairing constraint.
  (2) Genomic control: observed mean stem polymorphism vs matched-random relocations.
  (3) By regulatory context: is stem conservation stronger for promoter IRs?

Also reports per-IR "fragility" = fraction of accessions that disrupt the stem.

Outputs:
  reports/experiment-04-results.json
  reports/figures/exp04_stem_vs_loop.png
  reports/figures/exp04_conservation_null.png
  reports/figures/exp04_by_context.png

Run from repo root:  python experiments/ir_conservation_experiment.py
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
from scipy import stats

from genophenoir.config_loader import load_config, resolve_path
from genophenoir.ir_profiler import load_ir_bed
from genophenoir.ir_features import generate_matched_irs
from genophenoir.validation import _load_genotypes
from genophenoir.annotation import build_context_masks, assign_context, CONTEXTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp04")
sns.set_theme(style="whitegrid", context="notebook")


def stem_loop_snp_counts(ir, positions, alt_freq, poly_thresh=0.0):
    """Return (stem_sites, stem_len, loop_sites, loop_len, stem_mean_freq).

    stem_sites = number of polymorphic positions in the two stem arms;
    loop_sites = number in the loop/spacer. alt_freq is per-SNP frequency.
    """
    S = ir.stem_length
    # stem = [start, start+S) U [end-S, end);  loop = [start+S, end-S)
    def count(lo, hi):
        i0 = np.searchsorted(positions, lo, "left")
        i1 = np.searchsorted(positions, hi, "left")
        return i0, i1
    l0, l1 = count(ir.start, ir.start + S)
    r0, r1 = count(ir.end - S, ir.end)
    p0, p1 = count(ir.start + S, ir.end - S)
    stem_freqs = np.concatenate([alt_freq[l0:l1], alt_freq[r0:r1]])
    stem_sites = len(stem_freqs)
    loop_sites = p1 - p0
    loop_len = max(0, (ir.end - S) - (ir.start + S))
    return stem_sites, 2 * S, loop_sites, loop_len, (stem_freqs.mean() if stem_sites else 0.0)


def mean_stem_poly_rate(irs, positions):
    """Mean (polymorphic stem sites / stem length) across a set of IRs."""
    rates = []
    for ir in irs:
        S = ir.stem_length
        i0 = np.searchsorted(positions, ir.start, "left")
        i1 = np.searchsorted(positions, ir.start + S, "left")
        j0 = np.searchsorted(positions, ir.end - S, "left")
        j1 = np.searchsorted(positions, ir.end, "left")
        rates.append(((i1 - i0) + (j1 - j0)) / (2 * S))
    return float(np.mean(rates))


def main() -> None:
    cfg = load_config()
    chrom = cfg.params.chromosomes[0]
    reports = Path("reports"); figs = reports / "figures"; figs.mkdir(parents=True, exist_ok=True)

    irs = load_ir_bed(resolve_path(cfg, cfg.paths.phase1_output) / f"{chrom}_ir_regions.bed")
    variants_df, geno_acc = _load_genotypes(cfg, chrom)
    positions = variants_df["pos"].to_numpy()
    genotypes = variants_df[geno_acc].to_numpy(dtype=np.int8)
    alt_freq = genotypes.mean(axis=1)  # per-SNP alt-allele frequency
    chrom_length = int(positions.max()) + 1
    logger.info("IRs=%d, SNPs=%d, accessions=%d", len(irs), len(positions), len(geno_acc))

    # --- (1) Within-IR: stem vs loop polymorphism ---
    stem_rates, loop_rates, stem_freqs, fragility = [], [], [], []
    per_ir = []
    for ir in irs:
        ss, sl, ls, ll, smf = stem_loop_snp_counts(ir, positions, alt_freq)
        sr = ss / sl if sl else np.nan
        lr = ls / ll if ll else np.nan
        stem_rates.append(sr)
        loop_rates.append(lr)
        stem_freqs.append(smf)
        per_ir.append((ir, sr, lr))
    stem_rates = np.array(stem_rates); loop_rates = np.array(loop_rates)
    paired = ~np.isnan(stem_rates) & ~np.isnan(loop_rates)
    w_stat, w_p = stats.wilcoxon(stem_rates[paired], loop_rates[paired]) if paired.sum() > 10 else (np.nan, np.nan)
    logger.info("Stem poly rate=%.4f  Loop poly rate=%.4f  (Wilcoxon p=%.2e, n=%d)",
                np.nanmean(stem_rates), np.nanmean(loop_rates), w_p, int(paired.sum()))

    # --- (2) Genomic: observed mean stem poly vs matched-random relocations ---
    obs = mean_stem_poly_rate(irs, positions)
    rng = np.random.default_rng(11)
    null = np.array([mean_stem_poly_rate(generate_matched_irs(irs, chrom_length, rng), positions)
                     for _ in range(200)])
    p_cons = (1 + np.sum(null <= obs)) / (1 + len(null))  # one-sided: stems LESS polymorphic
    logger.info("Observed stem poly rate=%.4f  null=%.4f+/-%.4f  p(less polymorphic)=%.3f",
                obs, null.mean(), null.std(), p_cons)

    # --- (3) By regulatory context ---
    masks = build_context_masks(resolve_path(cfg, cfg.paths.gff3_file), chrom, chrom_length)
    ctx = np.array([assign_context((ir.start + ir.end) // 2, masks) for ir in irs])
    by_ctx = {}
    for c in CONTEXTS:
        sel = ctx == c
        if sel.sum() >= 30:
            by_ctx[c] = {"n": int(sel.sum()), "stem_poly_rate": float(np.nanmean(stem_rates[sel])),
                         "stem_mean_alt_freq": float(np.nanmean(np.array(stem_freqs)[sel]))}

    results = {
        "n_irs": len(irs),
        "stem_poly_rate_mean": float(np.nanmean(stem_rates)),
        "loop_poly_rate_mean": float(np.nanmean(loop_rates)),
        "wilcoxon_p_stem_vs_loop": float(w_p),
        "stem_lower_than_loop": bool(np.nanmean(stem_rates) < np.nanmean(loop_rates)),
        "observed_stem_poly_rate": obs,
        "null_stem_poly_mean": float(null.mean()),
        "null_stem_poly_std": float(null.std()),
        "p_stems_more_conserved_than_random": float(p_cons),
        "by_context": by_ctx,
    }
    (reports / "experiment-04-results.json").write_text(json.dumps(results, indent=2))
    logger.info("RESULTS:\n%s", json.dumps(results, indent=2))

    _fig_stem_vs_loop(stem_rates[paired], loop_rates[paired], figs / "exp04_stem_vs_loop.png")
    _fig_null(obs, null, figs / "exp04_conservation_null.png")
    _fig_context(by_ctx, figs / "exp04_by_context.png")
    logger.info("Figures written to %s", figs)
    print("EXPERIMENT_COMPLETE")


def _fig_stem_vs_loop(stem, loop, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([stem, loop], tick_labels=["Stem arms", "Loop/spacer"], showfliers=False,
               medianprops=dict(color="crimson"))
    ax.set_ylabel("Polymorphic-site rate (SNP sites / bp)")
    ax.set_title(f"Exp 04(1) — IR stem vs loop polymorphism\n"
                 f"stem={stem.mean():.3f}, loop={loop.mean():.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _fig_null(obs, null, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null, bins=25, color="0.7", edgecolor="white", label="Matched-random regions")
    ax.axvline(obs, color="crimson", lw=2, label=f"Observed IR stems ({obs:.3f})")
    p = (1 + np.sum(null <= obs)) / (1 + len(null))
    ax.set_xlabel("Mean stem polymorphism rate"); ax.set_ylabel("Count")
    ax.set_title(f"Exp 04(2) — Are IR stems less polymorphic than random?  p={p:.3f}")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _fig_context(by_ctx, path):
    names = list(by_ctx.keys())
    rates = [by_ctx[c]["stem_poly_rate"] for c in names]
    ns = [by_ctx[c]["n"] for c in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(names)), rates, color="#5b8db8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(names, ns)], rotation=20, fontsize=8)
    ax.set_ylabel("Mean stem polymorphism rate")
    ax.set_title("Exp 04(3) — IR stem conservation by regulatory context\n(lower = more conserved)")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
