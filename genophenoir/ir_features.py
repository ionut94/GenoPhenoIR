"""Mechanistic inverted-repeat features.

The first validation run showed the binary "any SNP in the region = disrupted"
feature carries no IR-specific signal — it is indistinguishable from random
matched regions. That feature throws away all IR biology: a SNP in the loop
disrupts nothing, while a SNP that breaks a stem base pair destabilises the
hairpin.

This module builds biology-aware features from the IR geometry + reference
sequence + per-accession genotypes, so the IR hypothesis can be tested fairly:

  - ``any_snp``  : binary, disrupted if >=1 SNP anywhere in [start, end)
                   (the original feature; kept as the baseline to beat).
  - ``stem_snp`` : binary, disrupted only if >=1 SNP falls in a *stem arm*
                   (loop/spacer SNPs ignored).
  - ``ddg``      : continuous destabilisation score — sum over disrupted stem
                   base pairs of a stability weight (G:C = 3, A:T = 2, else 1),
                   a ΔΔG-like proxy computable without folding every accession.

For an IR at [start, end) with stem length S, stem arm k (0..S-1) pairs the
left position ``start + k`` with the right position ``end - 1 - k``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from genophenoir.ir_profiler import InvertedRepeat

# Watson–Crick pair stability weights (H-bond count proxy). DNA, so T not U.
_PAIR_WEIGHT = {frozenset({"G", "C"}): 3.0, frozenset({"A", "T"}): 2.0}


def pair_weight(a: str, b: str) -> float:
    """Stability weight for a base pair (G:C=3, A:T=2, anything else=1)."""
    return _PAIR_WEIGHT.get(frozenset({a, b}), 1.0)


def generate_matched_irs(
    irs: list[InvertedRepeat],
    chrom_length: int,
    rng: np.random.Generator,
) -> list[InvertedRepeat]:
    """Relocate each IR to a random position, preserving its geometry.

    The matched null for mechanistic features: the IR structural template
    (stem length, spacer, total length) is held fixed and only its genomic
    location is randomised. This tests whether real IR *locations* carry more
    signal than random locations under the identical feature definition.
    """
    out: list[InvertedRepeat] = []
    for ir in irs:
        length = ir.end - ir.start
        if length <= 0 or length >= chrom_length:
            continue
        start = int(rng.integers(0, chrom_length - length))
        out.append(replace(ir, start=start, end=start + length))
    return out


def build_feature_fingerprints(
    irs: list[InvertedRepeat],
    positions: np.ndarray,
    genotypes: np.ndarray,
    reference: str,
    disruption_threshold: int = 1,
) -> dict[str, np.ndarray]:
    """Build all mechanistic features in a single pass over the IRs.

    Args:
        irs: Inverted repeats (with start/end/stem_length in chromosome coords).
        positions: Sorted 1D array of SNP positions (length = n_variants).
        genotypes: Binary matrix (n_variants x n_accessions), 0=ref, 1=alt.
        reference: Full chromosome reference sequence (upper-case).
        disruption_threshold: Min alt alleles to call a binary feature disrupted.

    Returns:
        Dict feature_name -> (n_accessions x n_irs) float32 matrix:
          - 'any_snp', 'stem_snp': 1 = intact, 0 = disrupted
          - 'ddg':                 0 = intact, positive = destabilisation score
    """
    n_acc = genotypes.shape[1]
    n_ir = len(irs)
    any_fp = np.ones((n_acc, n_ir), dtype=np.float32)
    stem_fp = np.ones((n_acc, n_ir), dtype=np.float32)
    ddg = np.zeros((n_acc, n_ir), dtype=np.float32)

    ref_len = len(reference)
    los = np.searchsorted(positions, [ir.start for ir in irs], side="left")
    his = np.searchsorted(positions, [ir.end for ir in irs], side="left")

    for j, ir in enumerate(irs):
        lo, hi = int(los[j]), int(his[j])
        if hi <= lo:
            continue
        block = genotypes[lo:hi, :]  # (m x n_acc)
        pos_slice = positions[lo:hi]

        # --- any-SNP (whole region) ---
        any_fp[block.sum(axis=0) >= disruption_threshold, j] = 0

        # --- stem-only + ΔΔG ---
        S = ir.stem_length
        left_lo, left_hi = ir.start, ir.start + S
        right_lo, right_hi = ir.end - S, ir.end
        if ir.end > ref_len:
            continue

        stem_alt_any = np.zeros(n_acc, dtype=bool)
        # Track which stem pairs an accession has disrupted (avoid double-count
        # when both arms of a pair carry a SNP) via per-pair weight then sum.
        disrupted = np.zeros((n_acc, S), dtype=bool)
        weights = np.zeros(S, dtype=np.float32)

        for r in range(hi - lo):
            p = int(pos_slice[r])
            if left_lo <= p < left_hi:
                k = p - ir.start
            elif right_lo <= p < right_hi:
                k = ir.end - 1 - p
            else:
                continue  # loop/spacer SNP — does not affect the hairpin stem
            if not (0 <= k < S):
                continue
            g = block[r, :] > 0  # accessions carrying the alt allele
            stem_alt_any |= g
            disrupted[:, k] |= g
            if weights[k] == 0:
                lp, rp = ir.start + k, ir.end - 1 - k
                weights[k] = pair_weight(reference[lp], reference[rp])

        stem_fp[stem_alt_any, j] = 0
        ddg[:, j] = (disrupted * weights[None, :]).sum(axis=1)

    return {"any_snp": any_fp, "stem_snp": stem_fp, "ddg": ddg}
