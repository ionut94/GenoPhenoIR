"""Shared configuration loader for all GenoPhenoIR modules."""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PathsConfig:
    reference_fasta: str
    phenotype_csv: str
    gff3_file: str
    output_base: str
    phase1_output: str
    phase2_output: str
    phase3_output: str
    phase4_output: str
    snp_matrix_hdf5: str = ""  # preferred genotype source (1001G imputed HDF5)
    vcf_file: str = ""  # legacy/alternative genotype source
    validation_output: str = "output/validation"
    kinship_hdf5: str = ""  # 1001G IBS kinship matrix (for structure correction)


@dataclass
class ParamsConfig:
    chromosomes: list[str] = field(default_factory=lambda: ["Chr1"])
    min_stem_length: int = 6
    max_stem_length: int = 100
    min_spacer_length: int = 0
    max_spacer_length: int = 50
    max_mismatches: int = 1
    min_ir_length: int = 14
    iupacpal_window_bp: int = 5_000_000
    max_accessions: int = 200
    ir_disruption_threshold: int = 1
    target_traits: list[str] = field(
        default_factory=lambda: ["FT16", "FT10"]
    )
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    hdbscan_min_cluster_size: int = 10
    cv_folds: int = 5
    shap_max_display: int = 20
    n_null_regions: int = 30
    n_structure_pcs: int = 10
    n_populations: int = 10
    validation_n_estimators: int = 100
    window_size_bp: int = 500_000
    image_height: int = 6
    latent_dim: int = 32
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 32
    autoencoder_lr: float = 0.001


@dataclass
class Config:
    paths: PathsConfig
    params: ParamsConfig
    project_root: Path


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A Config object with paths and params populated.
    """
    config_path = Path(config_path)
    project_root = config_path.parent.resolve()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    paths_cfg = PathsConfig(**raw["paths"])
    params_cfg = ParamsConfig(**raw["params"])

    return Config(paths=paths_cfg, params=params_cfg, project_root=project_root)


def resolve_path(cfg: Config, relative: str) -> Path:
    """Resolve a config-relative path to an absolute path."""
    return cfg.project_root / relative
