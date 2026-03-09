"""Phase 1 — Inverted Repeat Profiling.

Scans the TAIR10 reference genome for inverted repeats (palindromic sequences),
then overlays VCF variants to build a per-accession binary IR fingerprint matrix.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from genophenoir.config_loader import Config, load_config, resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InvertedRepeat:
    """A single inverted repeat locus on the reference genome."""

    chrom: str
    start: int
    end: int
    stem_length: int
    spacer_length: int
    sequence: str
    ir_id: str = ""

    @property
    def total_length(self) -> int:
        return self.end - self.start

    def to_bed_row(self) -> dict:
        return {
            "chrom": self.chrom,
            "start": self.start,
            "end": self.end,
            "ir_id": self.ir_id,
            "stem_length": self.stem_length,
            "spacer_length": self.spacer_length,
            "sequence": self.sequence,
        }


# ---------------------------------------------------------------------------
# IR detection — pure-Python palindrome scanner
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def scan_ir_python(
    sequence: str,
    chrom: str,
    min_stem: int = 6,
    max_stem: int = 100,
    min_spacer: int = 0,
    max_spacer: int = 50,
    max_mismatches: int = 1,
    min_total_length: int = 14,
) -> list[InvertedRepeat]:
    """Scan a DNA sequence for inverted repeats using a sliding-window approach.

    This is a simplified scanner suitable for proof-of-concept work. For each
    position, it checks whether a stem of length `s` followed by a spacer of
    length `d` is followed by the reverse complement of the stem (with up to
    `max_mismatches` allowed).

    For performance on chromosome-scale sequences, we use a coarse grid: we
    step through positions and stem/spacer combinations with stride to keep
    runtime manageable, then refine hits.

    Args:
        sequence: The DNA string (uppercase recommended).
        chrom: Chromosome name for output records.
        min_stem: Minimum stem arm length.
        max_stem: Maximum stem arm length.
        min_spacer: Minimum spacer (loop) length.
        max_spacer: Maximum spacer (loop) length.
        max_mismatches: Allowed mismatches between stem arms.
        min_total_length: Minimum total IR length to report.

    Returns:
        List of InvertedRepeat objects found.
    """
    seq_upper = sequence.upper()
    seq_len = len(seq_upper)
    results: list[InvertedRepeat] = []
    seen: set[tuple[int, int]] = set()  # (start, end) dedup

    # For PoC speed: sample stem lengths and use stride
    stem_lengths = list(range(min_stem, min(max_stem + 1, 31), 2))  # cap at 30 for speed
    spacer_lengths = list(range(min_spacer, min(max_spacer + 1, 21), 2))

    # Step through the sequence with a stride for speed
    stride = 50  # check every 50 bp; real tools check every position

    for pos in tqdm(
        range(0, seq_len - 2 * min_stem - min_spacer, stride),
        desc=f"Scanning {chrom}",
        leave=False,
    ):
        for stem_len in stem_lengths:
            for spacer_len in spacer_lengths:
                total = 2 * stem_len + spacer_len
                if total < min_total_length:
                    continue
                end_pos = pos + total
                if end_pos > seq_len:
                    continue

                left_arm = seq_upper[pos : pos + stem_len]
                right_arm_start = pos + stem_len + spacer_len
                right_arm = seq_upper[right_arm_start : right_arm_start + stem_len]

                # Check reverse complement match
                rc_left = reverse_complement(left_arm)
                mismatches = sum(
                    1 for a, b in zip(rc_left, right_arm) if a != b
                )

                if mismatches <= max_mismatches:
                    key = (pos, end_pos)
                    if key not in seen:
                        seen.add(key)
                        ir_seq = seq_upper[pos:end_pos]
                        results.append(
                            InvertedRepeat(
                                chrom=chrom,
                                start=pos,
                                end=end_pos,
                                stem_length=stem_len,
                                spacer_length=spacer_len,
                                sequence=ir_seq if len(ir_seq) <= 200 else ir_seq[:100] + "..." + ir_seq[-100:],
                            )
                        )

    # Sort by position
    results.sort(key=lambda ir: (ir.start, ir.end))

    # Assign IDs
    for i, ir in enumerate(results):
        ir.ir_id = f"{chrom}_IR_{i:06d}"

    logger.info("Found %d inverted repeats on %s", len(results), chrom)
    return results


# ---------------------------------------------------------------------------
# IR detection — iupacpal wrapper (optional, preferred if installed)
# ---------------------------------------------------------------------------

_IUPACPAL_SEARCH_PATHS = [
    "iupacpal",  # system PATH
    "tools/bin/iupacpal",  # project-local build
]


def _find_iupacpal() -> str | None:
    """Find the iupacpal binary, checking project-local and system PATH."""
    for candidate in _IUPACPAL_SEARCH_PATHS:
        try:
            result = subprocess.run(
                [candidate],
                capture_output=True,
                timeout=5,
            )
            # iupacpal prints help and exits with error when no input given
            return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def scan_ir_iupacpal(
    fasta_path: Path,
    chrom: str,
    seq_name: str | None = None,
    min_palindrome_len: int = 12,
    max_palindrome_len: int = 200,
    max_gap: int = 50,
    max_mismatches: int = 1,
) -> list[InvertedRepeat]:
    """Scan for inverted repeats using the IUPACpal external tool.

    IUPACpal CLI:
      -f input_file    FASTA file
      -s seq_name      Sequence name in FASTA header
      -m min_len       Minimum palindrome length (2*stem + spacer)
      -M max_len       Maximum palindrome length
      -g max_gap        Maximum spacer/gap between arms
      -x mismatches    Allowed mismatches
      -o output_file   Output file

    Falls back to the Python scanner if iupacpal is not installed.

    Args:
        fasta_path: Path to the FASTA file.
        chrom: Chromosome name for output records.
        seq_name: Sequence name in the FASTA header (auto-detected if None).
        min_palindrome_len: Minimum total palindrome length.
        max_palindrome_len: Maximum total palindrome length.
        max_gap: Maximum spacer/gap between arms.
        max_mismatches: Allowed mismatches between arms.

    Returns:
        List of InvertedRepeat objects.
    """
    iupacpal_bin = _find_iupacpal()
    if iupacpal_bin is None:
        logger.warning("iupacpal not found; falling back to Python scanner")
        seq_record = next(SeqIO.parse(fasta_path, "fasta"))
        return scan_ir_python(
            str(seq_record.seq),
            chrom,
            min_stem=min_palindrome_len // 2,
            max_stem=max_palindrome_len // 2,
            min_spacer=0,
            max_spacer=max_gap,
            max_mismatches=max_mismatches,
        )

    # Auto-detect sequence name from FASTA header if not provided
    if seq_name is None:
        seq_record = next(SeqIO.parse(fasta_path, "fasta"))
        seq_name = seq_record.id
        logger.info("Auto-detected sequence name: %s", seq_name)

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
        tmp_out = tmp.name

    cmd = [
        iupacpal_bin,
        "-f", str(fasta_path),
        "-s", seq_name,
        "-m", str(min_palindrome_len),
        "-M", str(max_palindrome_len),
        "-g", str(max_gap),
        "-x", str(max_mismatches),
        "-o", tmp_out,
    ]
    logger.info("Running IUPACpal: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error("IUPACpal failed: %s", result.stderr)
        # Fall back to Python scanner
        seq_record = next(SeqIO.parse(fasta_path, "fasta"))
        return scan_ir_python(
            str(seq_record.seq), chrom,
            min_stem=min_palindrome_len // 2,
            max_stem=max_palindrome_len // 2,
            min_spacer=0, max_spacer=max_gap,
            max_mismatches=max_mismatches,
        )

    results = _parse_iupacpal_output(tmp_out, chrom)
    Path(tmp_out).unlink(missing_ok=True)
    return results


def _parse_iupacpal_output(path: str, chrom: str) -> list[InvertedRepeat]:
    """Parse IUPACpal human-readable output into InvertedRepeat objects.

    IUPACpal output format (groups of 3 lines per palindrome):
        1        atgcatgca        9      ← left arm with start/end positions
                 |||||||||                ← match indicators
        22       tacgtacgt       14      ← right arm with start/end positions

    Positions are 1-based in the output.
    """
    results: list[InvertedRepeat] = []

    with open(path) as f:
        lines = f.readlines()

    # Find the "Palindromes:" header and skip everything before it
    palindrome_start = None
    for i, line in enumerate(lines):
        if line.strip() == "Palindromes:":
            palindrome_start = i + 1
            break

    if palindrome_start is None:
        logger.warning("No 'Palindromes:' section found in IUPACpal output")
        return results

    # Parse groups of 3 non-empty lines: left_arm, match_line, right_arm
    content_lines = []
    for line in lines[palindrome_start:]:
        stripped = line.strip()
        if stripped:
            content_lines.append(stripped)

    idx = 0
    ir_count = 0
    while idx + 2 < len(content_lines):
        left_line = content_lines[idx]
        _match_line = content_lines[idx + 1]  # pipe characters
        right_line = content_lines[idx + 2]
        idx += 3

        try:
            left_parts = left_line.split()
            right_parts = right_line.split()

            # Left arm: "start_pos  sequence  end_pos"
            left_start = int(left_parts[0])  # 1-based
            left_seq = left_parts[1]
            left_end = int(left_parts[2])  # 1-based

            # Right arm: "start_pos  sequence  end_pos"
            right_start = int(right_parts[0])  # 1-based
            right_seq = right_parts[1]
            right_end = int(right_parts[2])  # 1-based

            stem_length = len(left_seq)
            # Convert to 0-based coordinates
            ir_start = left_start - 1  # 0-based inclusive
            ir_end = right_start  # right_start is 1-based, so this is 0-based exclusive

            # Spacer length = gap between left arm end and right arm start
            spacer_length = ir_end - (ir_start + stem_length) - stem_length
            if spacer_length < 0:
                spacer_length = 0

            full_seq = left_seq
            if len(full_seq) > 200:
                full_seq = full_seq[:100] + "..." + full_seq[-100:]

            results.append(
                InvertedRepeat(
                    chrom=chrom,
                    start=ir_start,
                    end=ir_end,
                    stem_length=stem_length,
                    spacer_length=spacer_length,
                    sequence=full_seq,
                    ir_id=f"{chrom}_IR_{ir_count:06d}",
                )
            )
            ir_count += 1
        except (ValueError, IndexError) as e:
            logger.debug("Skipping unparseable palindrome lines: %s", e)
            continue

    logger.info("Parsed %d IRs from IUPACpal output", len(results))
    return results


# ---------------------------------------------------------------------------
# VCF overlay — build the IR fingerprint matrix
# ---------------------------------------------------------------------------

def load_vcf_variants(
    vcf_path: Path,
    chrom: str,
    max_accessions: int = 200,
) -> tuple[pd.DataFrame, list[str]]:
    """Load variant positions and genotypes from a VCF file.

    Uses pysam for indexed VCF access. Returns a DataFrame of variant
    positions and the list of sample (accession) names.

    Args:
        vcf_path: Path to a bgzipped + tabix-indexed VCF file.
        chrom: Chromosome to extract.
        max_accessions: Max number of accessions to load (for speed).

    Returns:
        Tuple of (variants_df, accession_ids).
        variants_df has columns: pos, ref, alt, and one column per accession
        with 0 (ref) or 1 (has alt allele).
    """
    import pysam

    vcf_path = Path(vcf_path)

    # Try tabix-indexed access first; fall back to full scan
    try:
        vcf = pysam.VariantFile(str(vcf_path))
    except Exception as e:
        logger.error("Cannot open VCF %s: %s", vcf_path, e)
        raise

    samples = list(vcf.header.samples)
    if max_accessions and len(samples) > max_accessions:
        samples = samples[:max_accessions]
        logger.info("Subsetting to %d accessions", max_accessions)

    rows = []
    try:
        variant_iter = vcf.fetch(chrom)
    except ValueError:
        # Chromosome name might not have 'Chr' prefix in VCF
        alt_chrom = chrom.replace("Chr", "")
        try:
            variant_iter = vcf.fetch(alt_chrom)
        except ValueError:
            logger.warning("Chromosome %s not found in VCF", chrom)
            return pd.DataFrame(), samples

    for rec in tqdm(variant_iter, desc=f"Loading VCF variants ({chrom})", leave=False):
        gt_row: dict = {"pos": rec.pos, "ref": rec.ref, "alt": str(rec.alts[0]) if rec.alts else "."}
        for s in samples:
            gt = rec.samples[s].get("GT", (0, 0))
            # 1 if any alt allele present
            gt_row[s] = 1 if gt and any(a and a > 0 for a in gt if a is not None) else 0
        rows.append(gt_row)

    variants_df = pd.DataFrame(rows)
    logger.info("Loaded %d variants for %d accessions on %s", len(variants_df), len(samples), chrom)
    return variants_df, samples


def build_ir_fingerprint(
    ir_list: list[InvertedRepeat],
    variants_df: pd.DataFrame,
    accessions: list[str],
    disruption_threshold: int = 1,
) -> pd.DataFrame:
    """Build a binary accession x IR fingerprint matrix.

    For each IR region, count how many variants an accession has within it.
    If the count >= disruption_threshold, mark as disrupted (0); otherwise
    intact (1).

    Args:
        ir_list: Inverted repeats found on the reference.
        variants_df: DataFrame from load_vcf_variants with 'pos' column
                     and one column per accession.
        accessions: List of accession IDs.
        disruption_threshold: Min variants to call an IR disrupted.

    Returns:
        DataFrame with accessions as rows and IR IDs as columns.
        Values: 1 = intact, 0 = disrupted.
    """
    if variants_df.empty:
        logger.warning("No variants loaded; returning all-intact fingerprint")
        return pd.DataFrame(
            1,
            index=accessions,
            columns=[ir.ir_id for ir in ir_list],
        )

    positions = variants_df["pos"].values
    fingerprint = {}

    for ir in tqdm(ir_list, desc="Building IR fingerprints", leave=False):
        # Find variants within this IR
        mask = (positions >= ir.start) & (positions < ir.end)
        if not mask.any():
            # No variants in this IR → intact for all accessions
            fingerprint[ir.ir_id] = {acc: 1 for acc in accessions}
            continue

        ir_variants = variants_df.loc[mask, accessions]
        # Sum variants per accession in this IR region
        var_counts = ir_variants.sum(axis=0)
        fingerprint[ir.ir_id] = {
            acc: 0 if var_counts[acc] >= disruption_threshold else 1
            for acc in accessions
        }

    fp_df = pd.DataFrame(fingerprint)
    fp_df.index.name = "accession_id"
    logger.info(
        "IR fingerprint matrix: %d accessions x %d IRs", fp_df.shape[0], fp_df.shape[1]
    )

    # Report summary stats
    disrupted_frac = (fp_df == 0).mean().mean()
    logger.info("Mean disruption rate across all IRs: %.2f%%", disrupted_frac * 100)

    return fp_df


# ---------------------------------------------------------------------------
# Save/load helpers
# ---------------------------------------------------------------------------

def save_ir_bed(ir_list: list[InvertedRepeat], output_path: Path) -> None:
    """Save IR list to a BED-like TSV file."""
    df = pd.DataFrame([ir.to_bed_row() for ir in ir_list])
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved %d IRs to %s", len(ir_list), output_path)


def load_ir_bed(path: Path) -> list[InvertedRepeat]:
    """Load IRs from a previously saved BED-like TSV file."""
    df = pd.read_csv(path, sep="\t")
    return [
        InvertedRepeat(
            chrom=row["chrom"],
            start=int(row["start"]),
            end=int(row["end"]),
            stem_length=int(row["stem_length"]),
            spacer_length=int(row["spacer_length"]),
            sequence=str(row["sequence"]),
            ir_id=str(row["ir_id"]),
        )
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase1(cfg: Config) -> tuple[list[InvertedRepeat], pd.DataFrame]:
    """Execute Phase 1: scan for IRs and build the fingerprint matrix.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Tuple of (ir_list, fingerprint_df).
    """
    out_dir = resolve_path(cfg, cfg.paths.phase1_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_irs: list[InvertedRepeat] = []
    all_fingerprints: list[pd.DataFrame] = []

    for chrom in cfg.params.chromosomes:
        logger.info("=== Processing %s ===", chrom)

        # --- IR detection ---
        bed_path = out_dir / f"{chrom}_ir_regions.bed"
        if bed_path.exists():
            logger.info("Loading cached IR regions from %s", bed_path)
            ir_list = load_ir_bed(bed_path)
        else:
            fasta_path = resolve_path(cfg, cfg.paths.reference_fasta)
            logger.info("Scanning %s for inverted repeats...", fasta_path)

            # Try IUPACpal first (faster, more accurate); fall back to Python
            iupacpal_bin = _find_iupacpal()
            if iupacpal_bin is not None:
                logger.info("Using IUPACpal for IR detection")
                min_pal_len = cfg.params.min_ir_length
                max_pal_len = 2 * cfg.params.max_stem_length + cfg.params.max_spacer_length
                ir_list = scan_ir_iupacpal(
                    fasta_path,
                    chrom,
                    min_palindrome_len=min_pal_len,
                    max_palindrome_len=max_pal_len,
                    max_gap=cfg.params.max_spacer_length,
                    max_mismatches=cfg.params.max_mismatches,
                )
            else:
                # Python scanner fallback
                logger.info("IUPACpal not found; using Python scanner")
                seq_record = None
                for rec in SeqIO.parse(fasta_path, "fasta"):
                    rec_name = rec.id.split()[0]
                    if rec_name == chrom or rec_name == chrom.replace("Chr", "") or f"Chr{rec_name}" == chrom:
                        seq_record = rec
                        break

                if seq_record is None:
                    logger.error("Chromosome %s not found in %s", chrom, fasta_path)
                    continue

                ir_list = scan_ir_python(
                    str(seq_record.seq),
                    chrom,
                    min_stem=cfg.params.min_stem_length,
                    max_stem=cfg.params.max_stem_length,
                    min_spacer=cfg.params.min_spacer_length,
                    max_spacer=cfg.params.max_spacer_length,
                    max_mismatches=cfg.params.max_mismatches,
                    min_total_length=cfg.params.min_ir_length,
                )
            save_ir_bed(ir_list, bed_path)

        all_irs.extend(ir_list)

        # --- VCF overlay ---
        fp_path = out_dir / f"{chrom}_fingerprint.parquet"
        if fp_path.exists():
            logger.info("Loading cached fingerprint from %s", fp_path)
            fp_df = pd.read_parquet(fp_path)
        else:
            vcf_path = resolve_path(cfg, cfg.paths.vcf_file)
            if vcf_path.exists():
                variants_df, accessions = load_vcf_variants(
                    vcf_path, chrom, max_accessions=cfg.params.max_accessions
                )
                fp_df = build_ir_fingerprint(
                    ir_list, variants_df, accessions,
                    disruption_threshold=cfg.params.ir_disruption_threshold,
                )
                fp_df.to_parquet(fp_path)
                logger.info("Saved fingerprint to %s", fp_path)
            else:
                logger.warning("VCF file not found at %s; generating synthetic fingerprint", vcf_path)
                fp_df = _generate_synthetic_fingerprint(ir_list, cfg.params.max_accessions)
                fp_df.to_parquet(fp_path)

        all_fingerprints.append(fp_df)

    # Combine fingerprints across chromosomes
    if all_fingerprints:
        combined_fp = pd.concat(all_fingerprints, axis=1)
    else:
        combined_fp = pd.DataFrame()

    combined_fp.to_parquet(out_dir / "combined_fingerprint.parquet")
    logger.info("Phase 1 complete. Total IRs: %d, fingerprint shape: %s", len(all_irs), combined_fp.shape)

    return all_irs, combined_fp


def _generate_synthetic_fingerprint(
    ir_list: list[InvertedRepeat],
    n_accessions: int,
) -> pd.DataFrame:
    """Generate a synthetic fingerprint matrix for testing without real VCF data.

    Creates realistic-looking binary data where most IRs are intact (1) with
    ~10-20% disruption rate, and some correlation structure.
    """
    rng = np.random.default_rng(42)
    n_irs = len(ir_list)

    # Base disruption probability per IR (varies by genomic region)
    ir_disruption_probs = rng.beta(1, 8, size=n_irs)  # mostly low disruption

    # Generate binary matrix
    matrix = np.zeros((n_accessions, n_irs), dtype=np.int8)
    for j in range(n_irs):
        matrix[:, j] = rng.binomial(1, 1 - ir_disruption_probs[j], size=n_accessions)

    accession_ids = [f"ACC_{i:04d}" for i in range(n_accessions)]
    ir_ids = [ir.ir_id for ir in ir_list]

    df = pd.DataFrame(matrix, index=accession_ids, columns=ir_ids)
    df.index.name = "accession_id"
    logger.info("Generated synthetic fingerprint: %d x %d", *df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config()
    run_phase1(cfg)
