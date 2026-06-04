"""Regulatory-context annotation of genomic positions from a TAIR10 GFF3.

Builds per-context coverage masks over a chromosome and assigns each position
(e.g. an IR midpoint) to one context by biological priority:
    promoter (<= PROMOTER_BP upstream of TSS) > 5'UTR > 3'UTR > exon/CDS
    > intron (inside a gene body but none of the above) > intergenic
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROMOTER_BP = 1000
CONTEXTS = ["promoter", "five_prime_UTR", "three_prime_UTR", "exon", "intron", "intergenic"]


def build_context_masks(gff_path: Path, chrom: str, chrom_len: int) -> dict[str, np.ndarray]:
    """Boolean coverage masks over the chromosome for each regulatory context.

    Args:
        gff_path: TAIR10 GFF3 file.
        chrom: Chromosome name (e.g. "Chr1"); the GFF uses the bare "1".
        chrom_len: Chromosome length in bp.

    Returns:
        Dict with boolean arrays for promoter / five_prime_UTR / three_prime_UTR
        / exon / gene_body (intron is derived as gene_body minus the others).
    """
    gff_chrom = chrom.replace("Chr", "")
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
    """Assign one regulatory context to a position by biological priority."""
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
