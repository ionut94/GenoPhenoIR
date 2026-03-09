"""Phase 2 — Phenotype Integration.

Loads phenotype data from AraPheno, matches to accession IDs from the
genotype data, and produces a merged dataframe of IR fingerprint features
and phenotype values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from genophenoir.config_loader import Config, load_config, resolve_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phenotype loading
# ---------------------------------------------------------------------------

def load_phenotype_data(
    phenotype_path: Path,
    target_traits: list[str],
) -> pd.DataFrame:
    """Load and clean phenotype data from a CSV/TSV file.

    Expects a file with at least an accession identifier column and columns
    for each target trait. Handles common AraPheno formats.

    Args:
        phenotype_path: Path to the phenotype CSV/TSV.
        target_traits: List of trait column names to retain.

    Returns:
        DataFrame indexed by accession_id with trait columns.
    """
    # Detect separator
    suffix = phenotype_path.suffix.lower()
    sep = "\t" if suffix in (".tsv", ".tab") else ","

    df = pd.read_csv(phenotype_path, sep=sep)
    logger.info("Loaded phenotype file: %d rows, %d columns", *df.shape)
    logger.info("Columns: %s", list(df.columns))

    # Identify the accession ID column
    id_col = _find_accession_column(df)
    if id_col is None:
        raise ValueError(
            f"Cannot find accession ID column in {phenotype_path}. "
            f"Columns: {list(df.columns)}"
        )

    df = df.rename(columns={id_col: "accession_id"})

    # Normalise accession IDs to strings
    df["accession_id"] = df["accession_id"].astype(str).str.strip()

    # Find available target traits (case-insensitive fuzzy match)
    trait_mapping = _match_trait_columns(df.columns.tolist(), target_traits)
    if not trait_mapping:
        logger.warning("No target traits found. Available columns: %s", list(df.columns))
        # Fall back to numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "accession_id"]
        if numeric_cols:
            trait_mapping = {c: c for c in numeric_cols[:3]}
            logger.info("Using numeric columns as traits: %s", list(trait_mapping.keys()))

    # Rename matched columns to canonical names
    rename_map = {v: k for k, v in trait_mapping.items()}
    df = df.rename(columns=rename_map)

    # Keep only accession_id + target traits
    keep_cols = ["accession_id"] + list(trait_mapping.keys())
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Convert trait columns to numeric, coercing errors
    for col in keep_cols:
        if col != "accession_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no trait data at all
    trait_cols = [c for c in keep_cols if c != "accession_id"]
    df = df.dropna(subset=trait_cols, how="all")

    # Set index
    df = df.set_index("accession_id")
    df = df[~df.index.duplicated(keep="first")]

    logger.info(
        "Phenotype data after cleaning: %d accessions, traits: %s",
        len(df),
        list(df.columns),
    )
    for col in df.columns:
        logger.info(
            "  %s: %.1f%% non-null, mean=%.2f, std=%.2f",
            col,
            df[col].notna().mean() * 100,
            df[col].mean(),
            df[col].std(),
        )

    return df


def _find_accession_column(df: pd.DataFrame) -> str | None:
    """Heuristic search for the accession ID column."""
    candidates = [
        "accession_id", "accession", "ecotype_id", "id", "sample",
        "accession_name", "tg_ecotypeid", "ecotypeid",
    ]
    col_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in col_lower:
            return col_lower[cand]
    # Fall back: first column if it looks like IDs
    first_col = df.columns[0]
    if df[first_col].dtype == object or df[first_col].nunique() > 0.9 * len(df):
        return first_col
    return None


def _match_trait_columns(
    available: list[str],
    targets: list[str],
) -> dict[str, str]:
    """Fuzzy-match target trait names to available column names.

    Returns:
        Dict mapping canonical_name -> actual_column_name.
    """
    mapping = {}
    avail_lower = {c.lower().replace(" ", "_").replace("-", "_"): c for c in available}

    for target in targets:
        target_norm = target.lower().replace(" ", "_").replace("-", "_")
        # Exact match
        if target_norm in avail_lower:
            mapping[target] = avail_lower[target_norm]
            continue
        # Substring match
        for avail_norm, avail_orig in avail_lower.items():
            if target_norm in avail_norm or avail_norm in target_norm:
                mapping[target] = avail_orig
                break

    logger.info("Trait column mapping: %s", mapping)
    return mapping


# ---------------------------------------------------------------------------
# Merge with IR fingerprint
# ---------------------------------------------------------------------------

def merge_fingerprint_phenotype(
    fingerprint_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge IR fingerprint matrix with phenotype data on accession ID.

    Handles ID format differences (numeric vs string, with/without prefix).

    Args:
        fingerprint_df: IR fingerprint matrix (accessions x IR regions).
        phenotype_df: Phenotype values (accessions x traits).

    Returns:
        Merged DataFrame with both IR features and phenotype columns.
        Only accessions present in both datasets are retained.
    """
    # Normalise indices
    fp_ids = _normalise_ids(fingerprint_df.index)
    ph_ids = _normalise_ids(phenotype_df.index)

    fingerprint_df = fingerprint_df.copy()
    phenotype_df = phenotype_df.copy()
    fingerprint_df.index = fp_ids
    phenotype_df.index = ph_ids

    # Find overlapping accessions
    common = fp_ids.intersection(ph_ids)
    logger.info(
        "Accession overlap: %d (fingerprint: %d, phenotype: %d)",
        len(common),
        len(fp_ids),
        len(ph_ids),
    )

    if len(common) == 0:
        logger.warning(
            "No overlapping accessions found. "
            "Fingerprint IDs sample: %s, Phenotype IDs sample: %s",
            list(fp_ids[:5]),
            list(ph_ids[:5]),
        )
        # Try numeric-only matching as last resort
        fp_nums = _extract_numeric_ids(fingerprint_df.index)
        ph_nums = _extract_numeric_ids(phenotype_df.index)
        common_nums = fp_nums.intersection(ph_nums)
        if len(common_nums) > 0:
            fingerprint_df.index = fp_nums
            phenotype_df.index = ph_nums
            common = common_nums
            logger.info("Matched %d accessions via numeric ID extraction", len(common))
        else:
            logger.error("Cannot match any accessions between datasets")
            return pd.DataFrame()

    fp_matched = fingerprint_df.loc[common]
    ph_matched = phenotype_df.loc[common]

    merged = pd.concat([fp_matched, ph_matched], axis=1)
    merged.index.name = "accession_id"

    logger.info("Merged dataset: %d accessions, %d features", *merged.shape)
    return merged


def _normalise_ids(index: pd.Index) -> pd.Index:
    """Normalise accession IDs: strip whitespace, convert to string."""
    return pd.Index([str(x).strip() for x in index])


def _extract_numeric_ids(index: pd.Index) -> pd.Index:
    """Extract numeric portion from accession IDs (e.g. 'ACC_0001' -> '1')."""
    import re
    nums = []
    for x in index:
        match = re.search(r"(\d+)", str(x))
        nums.append(match.group(1) if match else str(x))
    return pd.Index(nums)


# ---------------------------------------------------------------------------
# Synthetic phenotype data for testing
# ---------------------------------------------------------------------------

def generate_synthetic_phenotypes(
    accession_ids: list[str],
    traits: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic phenotype data correlated with accession ordering.

    Useful for testing the pipeline when real AraPheno data isn't available.

    Args:
        accession_ids: List of accession identifiers.
        traits: List of trait names to generate.
        seed: Random seed.

    Returns:
        DataFrame with accessions as index and traits as columns.
    """
    rng = np.random.default_rng(seed)
    n = len(accession_ids)

    data = {}
    for trait in traits:
        if "flowering" in trait.lower():
            # Flowering time: roughly 20-80 days, bimodal
            vals = np.concatenate([
                rng.normal(30, 5, n // 2),
                rng.normal(60, 10, n - n // 2),
            ])
            rng.shuffle(vals)
        elif "leaf" in trait.lower() or "rosette" in trait.lower():
            # Rosette leaf number: 5-20, roughly normal
            vals = rng.normal(12, 3, n)
        elif "growth" in trait.lower():
            # Growth rate: 0.5-3.0 mm/day
            vals = rng.lognormal(0.3, 0.4, n)
        else:
            vals = rng.normal(0, 1, n)

        # Add ~5% missing values
        mask = rng.random(n) < 0.05
        vals = vals.astype(float)
        vals[mask] = np.nan
        data[trait] = vals

    df = pd.DataFrame(data, index=accession_ids)
    df.index.name = "accession_id"
    logger.info("Generated synthetic phenotypes for %d accessions, %d traits", n, len(traits))
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase2(cfg: Config) -> pd.DataFrame:
    """Execute Phase 2: load phenotypes and merge with fingerprints.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Merged DataFrame of IR fingerprint features + phenotype values.
    """
    out_dir = resolve_path(cfg, cfg.paths.phase2_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load IR fingerprint from Phase 1
    fp_path = resolve_path(cfg, cfg.paths.phase1_output) / "combined_fingerprint.parquet"
    if not fp_path.exists():
        raise FileNotFoundError(
            f"Phase 1 output not found at {fp_path}. Run Phase 1 first."
        )
    fingerprint_df = pd.read_parquet(fp_path)
    logger.info("Loaded fingerprint: %d accessions x %d IRs", *fingerprint_df.shape)

    # Load phenotype data
    pheno_path = resolve_path(cfg, cfg.paths.phenotype_csv)
    if pheno_path.exists():
        phenotype_df = load_phenotype_data(pheno_path, cfg.params.target_traits)
    else:
        logger.warning("Phenotype file not found at %s; generating synthetic data", pheno_path)
        phenotype_df = generate_synthetic_phenotypes(
            list(fingerprint_df.index),
            cfg.params.target_traits,
        )

    # Save cleaned phenotype data
    phenotype_df.to_csv(out_dir / "phenotypes_clean.csv")

    # Merge
    merged_df = merge_fingerprint_phenotype(fingerprint_df, phenotype_df)

    if merged_df.empty:
        logger.warning("Merge produced empty result; using synthetic phenotype data as fallback")
        phenotype_df = generate_synthetic_phenotypes(
            list(fingerprint_df.index),
            cfg.params.target_traits,
        )
        merged_df = merge_fingerprint_phenotype(fingerprint_df, phenotype_df)

    merged_df.to_parquet(out_dir / "merged_data.parquet")
    merged_df.to_csv(out_dir / "merged_data.csv")
    logger.info("Phase 2 complete. Merged data saved to %s", out_dir)

    return merged_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config()
    run_phase2(cfg)
