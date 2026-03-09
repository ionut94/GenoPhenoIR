#!/usr/bin/env bash
# download_data.sh — Fetch Arabidopsis thaliana data for GenoPhenoIR
#
# Downloads:
#   1. TAIR10 reference genome (chromosome 1 FASTA)
#   2. 1001 Genomes VCF for chromosome 1
#   3. Sample phenotype data from AraPheno
#   4. TAIR10 GFF3 gene annotation
#
# Usage:
#   cd data/
#   bash download_data.sh
#
# Requirements: wget or curl, bgzip, tabix (from htslib)

set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DATA_DIR"

echo "=== GenoPhenoIR Data Download ==="
echo "Target directory: $DATA_DIR"
echo ""

# Helper: download with wget or curl
download() {
    local url="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "  Already exists: $output"
        return 0
    fi
    echo "  Downloading: $output"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$output" "$url"
    elif command -v curl &>/dev/null; then
        curl -L -o "$output" "$url"
    else
        echo "ERROR: Neither wget nor curl found. Please install one."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 1. TAIR10 Reference Genome — Chromosome 1
# ---------------------------------------------------------------------------
echo "--- 1. TAIR10 Reference Genome (Chr1) ---"

# Full genome from TAIR / Ensembl Plants
TAIR10_URL="https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr1.fas"
TAIR10_ALT_URL="https://ftp.ensemblgenomes.org/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa.gz"

if [ ! -f "TAIR10_chr1.fas" ]; then
    echo "  Trying primary source (TAIR)..."
    download "$TAIR10_URL" "TAIR10_chr1.fas" || {
        echo "  Primary source failed. Trying Ensembl Plants..."
        download "$TAIR10_ALT_URL" "TAIR10_chr1.fa.gz"
        gunzip -k "TAIR10_chr1.fa.gz" 2>/dev/null || true
        mv "TAIR10_chr1.fa" "TAIR10_chr1.fas" 2>/dev/null || \
        mv "Arabidopsis_thaliana.TAIR10.dna.chromosome.1.fa" "TAIR10_chr1.fas" 2>/dev/null || true
    }
fi
echo ""

# ---------------------------------------------------------------------------
# 2. 1001 Genomes VCF — Chromosome 1
# ---------------------------------------------------------------------------
echo "--- 2. 1001 Genomes VCF (Chr1) ---"

# The 1001 Genomes SNP matrix (full VCF is large; we download Chr1 subset)
# Source: https://1001genomes.org/data/GMI-MPI/releases/v3.1/
VCF_URL="https://1001genomes.org/data/GMI-MPI/releases/v3.1/1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz"

echo "  NOTE: The full 1001 Genomes VCF is ~2 GB compressed."
echo "  For the proof-of-concept, we recommend downloading and subsetting Chr1."
echo ""

if [ ! -f "1001genomes_chr1.vcf.gz" ]; then
    if [ -f "1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz" ]; then
        echo "  Full VCF already downloaded. Subsetting chromosome 1..."
    else
        echo "  Downloading full VCF (this may take a while)..."
        download "$VCF_URL" "1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz" || {
            echo ""
            echo "  ALTERNATIVE: If automatic download fails, manually download from:"
            echo "    https://1001genomes.org/data/GMI-MPI/releases/v3.1/"
            echo "  Place the file as: 1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz"
            echo ""
        }
    fi

    # Index and subset if htslib tools available
    if command -v tabix &>/dev/null; then
        if [ -f "1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz" ]; then
            # Index if needed
            if [ ! -f "1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz.tbi" ]; then
                echo "  Indexing VCF with tabix..."
                tabix -p vcf "1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz"
            fi
            # Extract Chr1
            echo "  Extracting chromosome 1..."
            tabix -h "1001genomes_snp-short-indel_only_ACGTN_v3.1.vcf.gz" 1 | \
                bgzip > "1001genomes_chr1.vcf.gz"
            tabix -p vcf "1001genomes_chr1.vcf.gz"
            echo "  Created: 1001genomes_chr1.vcf.gz"
        fi
    else
        echo "  WARNING: tabix not found. Install htslib to subset the VCF."
        echo "  You can install it with: brew install htslib (macOS) or apt install tabix (Linux)"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 3. AraPheno Phenotype Data
# ---------------------------------------------------------------------------
echo "--- 3. AraPheno Phenotype Data ---"

# AraPheno REST API for specific studies
# Study 3: Atwell et al. 2010 — flowering time, growth traits
ARAPHENO_URL="https://arapheno.1001genomes.org/rest/study/6/phenotypes.csv"
ARAPHENO_ALT="https://arapheno.1001genomes.org/rest/phenotype/flowering_time/values.csv"

if [ ! -f "arapheno_phenotypes.csv" ]; then
    echo "  Downloading phenotype data from AraPheno..."
    download "$ARAPHENO_URL" "arapheno_phenotypes.csv" || {
        echo "  Primary URL failed. Trying alternative..."
        download "$ARAPHENO_ALT" "arapheno_phenotypes.csv" || {
            echo ""
            echo "  Auto-download failed. Creating a sample phenotype file for testing."
            echo "  For real data, visit: https://arapheno.1001genomes.org/"
            echo "  and download flowering time / growth rate phenotypes."
            echo ""
            _create_sample_phenotypes
        }
    }
fi
echo ""

# Create sample phenotypes if download failed
_create_sample_phenotypes() {
    cat > "arapheno_phenotypes.csv" << 'PHENOEOF'
accession_id,flowering_time_LD,rosette_leaf_number,growth_rate
6909,25.3,8.2,1.45
6911,31.7,10.1,1.22
6915,28.4,9.3,1.38
6918,45.2,14.7,0.98
6923,22.1,7.5,1.67
PHENOEOF
    echo "  Created sample phenotype file (5 accessions for testing)"
}

# ---------------------------------------------------------------------------
# 4. TAIR10 GFF3 Gene Annotation
# ---------------------------------------------------------------------------
echo "--- 4. TAIR10 GFF3 Gene Annotation ---"

GFF3_URL="https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff"
GFF3_ALT_URL="https://ftp.ensemblgenomes.org/pub/plants/release-57/gff3/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.57.gff3.gz"

if [ ! -f "TAIR10_GFF3_genes.gff" ]; then
    download "$GFF3_URL" "TAIR10_GFF3_genes.gff" || {
        echo "  Primary source failed. Trying Ensembl Plants..."
        download "$GFF3_ALT_URL" "TAIR10_GFF3_genes.gff.gz"
        gunzip "TAIR10_GFF3_genes.gff.gz"
        mv "Arabidopsis_thaliana.TAIR10.57.gff3" "TAIR10_GFF3_genes.gff" 2>/dev/null || true
    }
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Download Summary ==="
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.{fas,fa,vcf.gz,csv,gff}* 2>/dev/null || echo "  (some files may be missing)"
echo ""
echo "If any downloads failed, see the README.md for manual download instructions."
echo "For the proof-of-concept, the pipeline can generate synthetic data as fallback."
echo "=== Done ==="
