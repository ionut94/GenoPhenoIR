#!/usr/bin/env bash
# download_data.sh — Fetch Arabidopsis thaliana data for GenoPhenoIR
#
# Downloads (all verified working as of 2026-06):
#   1. TAIR10 reference genome, chromosome 1 (Ensembl Plants, ~9 MB gz -> 30 MB)
#   2. 1001 Genomes imputed SNP matrix, HDF5 (~332 MB tar.gz, all 1135 accessions)
#   3. AraPheno study 12 flowering-time phenotypes (FT10/FT16, ~27 KB)
#   4. TAIR10 GFF3 gene annotation (optional, for SHAP gene overlap)
#
# Usage:
#   cd data/
#   bash download_data.sh
#
# Requirements: curl (or wget). No bgzip/tabix needed for the HDF5 path.

set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DATA_DIR"

echo "=== GenoPhenoIR Data Download ==="
echo "Target directory: $DATA_DIR"
echo ""

download() {
    local url="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "  Already exists: $output"
        return 0
    fi
    echo "  Downloading: $output"
    if command -v curl &>/dev/null; then
        curl -fSL --retry 3 -o "$output" "$url"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$output" "$url"
    else
        echo "ERROR: Neither curl nor wget found." >&2
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. TAIR10 Reference Genome — Chromosome 1
# ---------------------------------------------------------------------------
echo "--- 1. TAIR10 Reference Genome (Chr1) ---"
# NOTE: Ensembl Plants https cert is currently broken for this host; use http.
TAIR10_URL="http://ftp.ensemblgenomes.org/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz"
if [ ! -f "TAIR10_chr1.fas" ]; then
    download "$TAIR10_URL" "TAIR10_chr1.fa.gz"
    gunzip -kf "TAIR10_chr1.fa.gz"
    mv -f "TAIR10_chr1.fa" "TAIR10_chr1.fas"
    rm -f "TAIR10_chr1.fa.gz"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. 1001 Genomes Imputed SNP Matrix (HDF5) — preferred genotype source
# ---------------------------------------------------------------------------
echo "--- 2. 1001 Genomes Imputed SNP Matrix (HDF5) ---"
# Full genome, all 1135 accessions, no missing genotypes. ~332 MB compressed.
# (The full v3.1 VCF is ~19 GB and is NOT recommended for a laptop.)
SNP_MATRIX_URL="https://1001genomes.org/data/GMI-MPI/releases/v3.1/SNP_matrix_imputed_hdf5/1001_SNP_MATRIX.tar.gz"
if [ ! -f "1001_SNP_MATRIX/imputed_snps_binary.hdf5" ]; then
    download "$SNP_MATRIX_URL" "1001_SNP_MATRIX.tar.gz"
    echo "  Extracting..."
    tar xzf "1001_SNP_MATRIX.tar.gz"
    rm -f "1001_SNP_MATRIX.tar.gz"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. AraPheno Phenotype Data — study 12 (flowering time FT10/FT16)
# ---------------------------------------------------------------------------
echo "--- 3. AraPheno Flowering-Time Phenotypes (study 12) ---"
ARAPHENO_URL="https://arapheno.1001genomes.org/rest/study/12/values.csv"
download "$ARAPHENO_URL" "arapheno_flowering_time.csv"
echo ""

# ---------------------------------------------------------------------------
# 4. TAIR10 GFF3 Gene Annotation (optional — used for SHAP gene overlap)
# ---------------------------------------------------------------------------
echo "--- 4. TAIR10 GFF3 Gene Annotation ---"
GFF3_URL="http://ftp.ensemblgenomes.org/pub/plants/release-57/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.57.gff3.gz"
if [ ! -f "TAIR10_GFF3_genes.gff" ]; then
    download "$GFF3_URL" "TAIR10_GFF3_genes.gff.gz" || {
        echo "  GFF3 download failed (optional — pipeline runs without it)."
    }
    [ -f "TAIR10_GFF3_genes.gff.gz" ] && gunzip -f "TAIR10_GFF3_genes.gff.gz" && \
        mv -f "Arabidopsis_thaliana.TAIR10.57.gff3" "TAIR10_GFF3_genes.gff" 2>/dev/null || true
fi
echo ""

# ---------------------------------------------------------------------------
echo "=== Download Summary ==="
ls -lh "$DATA_DIR" 2>/dev/null
echo ""
echo "Done. Run the pipeline with: python -m genophenoir.ir_profiler"
