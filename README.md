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
| 3b | `validation.py` | **Matched-null + kinship correction** — is the signal IR-specific and beyond population structure? |
| 4 | `image_encoder.py` | 2D image encoding, CNN autoencoder, latent space comparison |

## Data Sources

- **Reference genome**: TAIR10 Chr1 (Ensembl Plants release-57)
- **Variants**: 1001 Genomes Project **imputed SNP matrix v3.1 (HDF5)** — full genome, all 1135 accessions, no missing genotypes (~332 MB)
- **Phenotypes**: AraPheno study 12 — flowering time FT10/FT16 (~1160 accessions)
- **Annotations**: TAIR10 GFF3 gene models

> The genotype source is the imputed HDF5 SNP matrix, not the raw VCF. The full
> v3.1 VCF is ~19 GB and impractical for local work; the HDF5 matrix is the
> recommended path and is read directly by `ir_profiler.load_hdf5_variants`.

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
| TAIR10 Chr1 FASTA | `http://ftp.ensemblgenomes.org/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz` (http only — cert broken on https) | `data/TAIR10_chr1.fas` |
| 1001 Genomes imputed SNP matrix | [`1001_SNP_MATRIX.tar.gz`](https://1001genomes.org/data/GMI-MPI/releases/v3.1/SNP_matrix_imputed_hdf5/1001_SNP_MATRIX.tar.gz) (~332 MB) | extract to `data/1001_SNP_MATRIX/imputed_snps_binary.hdf5` |
| AraPheno phenotypes | [`study/12/values.csv`](https://arapheno.1001genomes.org/rest/study/12/values.csv) | `data/arapheno_flowering_time.csv` |
| TAIR10 GFF3 | `http://ftp.ensemblgenomes.org/pub/plants/release-57/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.57.gff3.gz` | `data/TAIR10_GFF3_genes.gff` |

**Note:** If genotype/phenotype files are missing, the pipeline falls back to
**synthetic data** so it can run end-to-end — but synthetic phenotypes are random
and uncorrelated with the fingerprint, so a synthetic run demonstrates plumbing
only, not biological signal.

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

# Phase 3b: Validation — does the IR signal survive matched-null + kinship correction?
# (needs only the Phase 1 IR BED; builds its own fingerprints efficiently)
python -m genophenoir.validation

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
