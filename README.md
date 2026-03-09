# GenoPhenoIR

**Inverted Repeat Profiling and Phenotype Prediction for Plant Genomes**

A proof-of-concept pipeline exploring whether inverted repeat (IR) patterns in *Arabidopsis thaliana* genomes, combined with phenotypic data, can be used for pattern-based clustering and prediction relevant to plant breeding.

## Rationale

Inverted repeats (IRs) — palindromic sequences where one strand reads the same as its complement in reverse — are abundant structural features in plant genomes. They can form hairpin/cruciform structures, influence local chromatin architecture, and potentially affect gene regulation. This pipeline investigates whether the pattern of IR variation across accessions correlates with observable phenotypic differences, using the well-characterised *Arabidopsis thaliana* 1001 Genomes dataset.

The approach:
1. Detect IRs on the reference genome
2. Overlay population-level variants (SNPs) to determine which IRs are intact vs disrupted per accession
3. Use the resulting binary "IR fingerprint" matrix for unsupervised clustering and supervised phenotype prediction
4. Explore whether a 2D image encoding of IR profiles captures additional information via deep learning

## Architecture

The pipeline has 4 phases, implemented as independent Python modules:

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `ir_profiler.py` | Scan reference genome for IRs; build per-accession fingerprint from VCF |
| 2 | `phenotype_loader.py` | Load and merge AraPheno phenotype data with IR fingerprints |
| 3 | `pattern_analysis.py` | UMAP/HDBSCAN clustering, RF/GBR prediction, SHAP explainability |
| 4 | `image_encoder.py` | 2D image encoding, CNN autoencoder, latent space comparison |

## Data Sources

- **Reference genome**: TAIR10 (The Arabidopsis Information Resource)
- **Variants**: 1001 Genomes Project SNP matrix v3.1
- **Phenotypes**: AraPheno database
- **Annotations**: TAIR10 GFF3 gene models

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Download data

```bash
cd data/
bash download_data.sh
cd ..
```

If automatic downloads fail, manually obtain:

| File | Source | Place as |
|------|--------|----------|
| TAIR10 Chr1 FASTA | [TAIR](https://www.arabidopsis.org/) or [Ensembl Plants](https://plants.ensembl.org/) | `data/TAIR10_chr1.fas` |
| 1001 Genomes VCF | [1001genomes.org](https://1001genomes.org/data/GMI-MPI/releases/v3.1/) | `data/1001genomes_chr1.vcf.gz` (+ .tbi index) |
| AraPheno phenotypes | [AraPheno](https://arapheno.1001genomes.org/) | `data/arapheno_phenotypes.csv` |
| TAIR10 GFF3 | [TAIR](https://www.arabidopsis.org/) | `data/TAIR10_GFF3_genes.gff` |

**Note:** The pipeline generates synthetic data as a fallback when real data files are not available, so you can test the full pipeline without downloading anything.

### 3. (Optional) Install iupacpal for faster IR detection

```bash
git clone https://github.com/steven31415/iupacpal.git
cd iupacpal && make && sudo cp iupacpal /usr/local/bin/
```

The pipeline falls back to a pure-Python IR scanner if iupacpal is not installed.

## Running the Pipeline

### Run each phase independently

```bash
# Phase 1: IR profiling (run once; results are cached)
python -m genophenoir.ir_profiler

# Phase 2: Phenotype integration
python -m genophenoir.phenotype_loader

# Phase 3: Pattern discovery
python -m genophenoir.pattern_analysis

# Phase 4: Image encoding (requires PyTorch)
python -m genophenoir.image_encoder
```

### Run the full pipeline via Jupyter

```bash
jupyter notebook notebooks/exploration.ipynb
```

## Configuration

All paths and parameters are in `config.yaml`. Key settings:

- `params.chromosomes`: Which chromosomes to process (default: Chr1 only)
- `params.max_accessions`: Limit accessions for speed (default: 200)
- `params.min_stem_length` / `max_stem_length`: IR detection parameters
- `params.cv_folds`: Cross-validation folds for prediction

## Output

Results are saved under `output/`:

```
output/
├── phase1/
│   ├── Chr1_ir_regions.bed          # IR positions on reference
│   ├── Chr1_fingerprint.parquet     # Per-accession IR variation
│   └── combined_fingerprint.parquet
├── phase2/
│   ├── phenotypes_clean.csv
│   └── merged_data.parquet          # IR features + phenotype values
├── phase3/
│   ├── umap_clusters.csv/.png       # UMAP + HDBSCAN clustering
│   ├── pred_*.png                   # Prediction scatter plots
│   ├── prediction_summary.csv       # Model performance metrics
│   ├── shap_summary_*.png           # SHAP feature importance
│   └── shap_top_features_*.csv      # Top features with genomic coords
└── phase4/
    ├── images/                      # Per-accession PNG images
    ├── autoencoder_loss.png         # Training curve
    ├── latent_vectors.csv           # Autoencoder embeddings
    ├── latent_umap_*.png            # Latent space visualisations
    └── comparison_tab_vs_img.png    # Phase 3 vs Phase 4 comparison
```

## Dependencies

Key libraries: biopython, pandas, numpy, scikit-learn, umap-learn, hdbscan, shap, matplotlib, seaborn, pysam, torch, pyyaml, tqdm.

Full list in `pyproject.toml`.

## Limitations

This is an exploratory proof-of-concept:
- IR detection uses a simplified scanner with stride (not every position checked)
- Only chromosome 1 and 200 accessions by default
- Phenotype prediction accuracy depends heavily on the biological signal present
- The image encoding approach is experimental
