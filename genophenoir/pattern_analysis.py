"""Phase 3 — Pattern Discovery.

Clustering (UMAP + HDBSCAN), predictive modelling (RF / GBR with cross-validation),
and explainability (SHAP) on the IR fingerprint × phenotype dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import umap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from genophenoir.config_loader import Config, load_config, resolve_path

logger = logging.getLogger(__name__)

# Consistent plot style
sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_features_targets(
    merged_df: pd.DataFrame,
    target_traits: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the merged dataframe into IR features (X) and phenotype targets (y).

    Args:
        merged_df: Combined IR fingerprint + phenotype data.
        target_traits: Trait column names.

    Returns:
        Tuple of (X features, y targets).
    """
    trait_cols = [c for c in target_traits if c in merged_df.columns]
    feature_cols = [c for c in merged_df.columns if c not in trait_cols]

    X = merged_df[feature_cols].copy()
    y = merged_df[trait_cols].copy()

    logger.info("Features: %d columns, Targets: %s", X.shape[1], trait_cols)
    return X, y


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> pd.DataFrame:
    """Apply UMAP dimensionality reduction and HDBSCAN clustering.

    Generates cluster visualisations coloured by phenotype values.

    Args:
        X: IR fingerprint feature matrix.
        y: Phenotype target values.
        cfg: Pipeline configuration.
        out_dir: Directory to save plots.

    Returns:
        DataFrame with UMAP coordinates and cluster assignments.
    """
    logger.info("Running UMAP (n_neighbors=%d, min_dist=%.2f)...",
                cfg.params.umap_n_neighbors, cfg.params.umap_min_dist)

    # Remove zero-variance features
    var = X.var()
    X_filtered = X.loc[:, var > 0]
    logger.info("Features after variance filter: %d -> %d", X.shape[1], X_filtered.shape[1])

    if X_filtered.shape[1] < 2:
        logger.warning("Too few variable features for UMAP; using raw features")
        X_filtered = X

    reducer = umap.UMAP(
        n_neighbors=cfg.params.umap_n_neighbors,
        min_dist=cfg.params.umap_min_dist,
        n_components=cfg.params.umap_n_components,
        random_state=42,
        metric="jaccard",  # appropriate for binary features
    )
    embedding = reducer.fit_transform(X_filtered.values)

    umap_df = pd.DataFrame(
        embedding,
        columns=["UMAP1", "UMAP2"],
        index=X.index,
    )

    # HDBSCAN clustering
    logger.info("Running HDBSCAN (min_cluster_size=%d)...", cfg.params.hdbscan_min_cluster_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.params.hdbscan_min_cluster_size,
        metric="euclidean",
    )
    umap_df["cluster"] = clusterer.fit_predict(embedding)
    n_clusters = umap_df["cluster"].nunique() - (1 if -1 in umap_df["cluster"].values else 0)
    n_noise = (umap_df["cluster"] == -1).sum()
    logger.info("Found %d clusters, %d noise points", n_clusters, n_noise)

    # Save cluster assignments
    umap_df.to_csv(out_dir / "umap_clusters.csv")

    # Visualise: plain clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        umap_df["UMAP1"], umap_df["UMAP2"],
        c=umap_df["cluster"], cmap="tab20", s=15, alpha=0.7,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("IR Fingerprint Clusters (UMAP + HDBSCAN)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    fig.savefig(out_dir / "umap_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Visualise: colour by each phenotype
    for trait in y.columns:
        trait_vals = y[trait].reindex(umap_df.index)
        mask = trait_vals.notna()
        if mask.sum() < 10:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            umap_df.loc[mask, "UMAP1"],
            umap_df.loc[mask, "UMAP2"],
            c=trait_vals[mask],
            cmap="viridis",
            s=15,
            alpha=0.7,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"IR Clusters coloured by {trait}")
        plt.colorbar(scatter, ax=ax, label=trait)
        fig.savefig(
            out_dir / f"umap_{trait}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        logger.info("Saved cluster plot for trait: %s", trait)

    return umap_df


# ---------------------------------------------------------------------------
# Predictive modelling
# ---------------------------------------------------------------------------

def run_prediction(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> dict[str, dict]:
    """Train RF and GBR models to predict phenotype from IR features.

    Uses k-fold cross-validation and compares against a shuffled baseline.

    Args:
        X: IR fingerprint feature matrix.
        y: Phenotype target values.
        cfg: Pipeline configuration.
        out_dir: Directory to save results.

    Returns:
        Dict mapping trait -> model performance metrics.
    """
    results = {}

    for trait in y.columns:
        logger.info("--- Predicting: %s ---", trait)
        mask = y[trait].notna()
        X_t = X.loc[mask]
        y_t = y.loc[mask, trait]

        if len(y_t) < 20:
            logger.warning("Skipping %s: only %d samples", trait, len(y_t))
            continue

        # Remove zero-variance features
        var = X_t.var()
        X_t = X_t.loc[:, var > 0]

        trait_results = {}

        for model_name, model_cls in [
            ("RandomForest", RandomForestRegressor),
            ("GradientBoosting", GradientBoostingRegressor),
        ]:
            model = model_cls(
                n_estimators=200,
                random_state=42,
                **({"n_jobs": -1} if model_name == "RandomForest" else {}),
            )

            # Cross-validated predictions
            cv_scores = cross_val_score(
                model, X_t, y_t,
                cv=cfg.params.cv_folds,
                scoring="r2",
            )
            cv_preds = cross_val_predict(
                model, X_t, y_t,
                cv=cfg.params.cv_folds,
            )

            r2 = r2_score(y_t, cv_preds)
            corr, pval = pearsonr(y_t, cv_preds)

            # Shuffled baseline
            rng = np.random.default_rng(42)
            y_shuffled = y_t.values.copy()
            rng.shuffle(y_shuffled)
            baseline_scores = cross_val_score(
                model_cls(n_estimators=100, random_state=42,
                          **({"n_jobs": -1} if model_name == "RandomForest" else {})),
                X_t, y_shuffled,
                cv=cfg.params.cv_folds,
                scoring="r2",
            )

            trait_results[model_name] = {
                "r2": r2,
                "r2_cv_mean": cv_scores.mean(),
                "r2_cv_std": cv_scores.std(),
                "pearson_r": corr,
                "pearson_pval": pval,
                "baseline_r2_mean": baseline_scores.mean(),
                "baseline_r2_std": baseline_scores.std(),
            }

            logger.info(
                "%s | %s: R²=%.4f (CV: %.4f±%.4f), r=%.4f (p=%.2e) | Baseline R²=%.4f±%.4f",
                trait, model_name, r2,
                cv_scores.mean(), cv_scores.std(),
                corr, pval,
                baseline_scores.mean(), baseline_scores.std(),
            )

            # Prediction vs actual scatter
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_t, cv_preds, alpha=0.5, s=15)
            lims = [
                min(y_t.min(), cv_preds.min()),
                max(y_t.max(), cv_preds.max()),
            ]
            ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect prediction")
            ax.set_xlabel(f"Actual {trait}")
            ax.set_ylabel(f"Predicted {trait}")
            ax.set_title(f"{model_name}: {trait} (R²={r2:.3f}, r={corr:.3f})")
            ax.legend()
            fig.savefig(
                out_dir / f"pred_{trait}_{model_name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

        results[trait] = trait_results

    # Save summary
    summary_rows = []
    for trait, models in results.items():
        for model_name, metrics in models.items():
            summary_rows.append({"trait": trait, "model": model_name, **metrics})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "prediction_summary.csv", index=False)
    logger.info("Prediction summary:\n%s", summary_df.to_string())

    return results


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def run_shap_analysis(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
    gff3_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply SHAP TreeExplainer to identify important IR features.

    Trains the best model on full data, computes SHAP values, generates
    summary and beeswarm plots, and maps top features to genomic coordinates.

    Args:
        X: IR fingerprint feature matrix.
        y: Phenotype target values.
        cfg: Pipeline configuration.
        out_dir: Directory to save plots and results.
        gff3_path: Optional path to GFF3 annotation for gene overlap.

    Returns:
        Dict mapping trait -> DataFrame of top SHAP features with genomic info.
    """
    shap_results = {}

    for trait in y.columns:
        logger.info("--- SHAP analysis for: %s ---", trait)
        mask = y[trait].notna()
        X_t = X.loc[mask]
        y_t = y.loc[mask, trait]

        if len(y_t) < 20:
            continue

        var = X_t.var()
        X_t = X_t.loc[:, var > 0]

        # Train model on all data for SHAP
        model = GradientBoostingRegressor(
            n_estimators=200, random_state=42, max_depth=5
        )
        model.fit(X_t, y_t)

        # SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_t)

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_t,
            max_display=cfg.params.shap_max_display,
            show=False,
        )
        plt.title(f"SHAP Summary: {trait}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_{trait}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X_t.values,
                feature_names=X_t.columns.tolist(),
            ),
            max_display=cfg.params.shap_max_display,
            show=False,
        )
        plt.title(f"SHAP Beeswarm: {trait}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_beeswarm_{trait}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][: cfg.params.shap_max_display]
        top_features = pd.DataFrame({
            "ir_id": X_t.columns[top_idx],
            "mean_abs_shap": mean_abs_shap[top_idx],
        })

        # Parse genomic coordinates from IR IDs
        top_features["chrom"] = top_features["ir_id"].str.extract(r"^(Chr\w+)_IR_")
        top_features["ir_index"] = top_features["ir_id"].str.extract(r"_IR_(\d+)$").astype(int)

        # Attempt gene overlap if GFF3 available
        if gff3_path and gff3_path.exists():
            top_features = _annotate_gene_overlap(top_features, gff3_path, cfg)

        top_features.to_csv(out_dir / f"shap_top_features_{trait}.csv", index=False)
        shap_results[trait] = top_features
        logger.info("Top 5 SHAP features for %s:\n%s", trait, top_features.head().to_string())

    return shap_results


def _annotate_gene_overlap(
    features_df: pd.DataFrame,
    gff3_path: Path,
    cfg: Config,
) -> pd.DataFrame:
    """Check if top IR features overlap with annotated genes.

    Loads IR BED file from Phase 1 to get coordinates, then checks against
    the GFF3 annotation.

    Args:
        features_df: DataFrame with ir_id column.
        gff3_path: Path to TAIR10 GFF3 file.
        cfg: Pipeline configuration.

    Returns:
        features_df with added 'overlapping_genes' column.
    """
    # Load gene annotations from GFF3
    genes = _load_gff3_genes(gff3_path)
    if genes.empty:
        features_df["overlapping_genes"] = ""
        return features_df

    # Load IR coordinates from Phase 1
    from genophenoir.ir_profiler import load_ir_bed

    overlapping = []
    for _, row in features_df.iterrows():
        chrom = row.get("chrom", "")
        ir_id = row["ir_id"]

        # Try to load the IR BED to get coordinates
        bed_path = resolve_path(cfg, cfg.paths.phase1_output) / f"{chrom}_ir_regions.bed"
        if bed_path.exists():
            ir_list = load_ir_bed(bed_path)
            ir_dict = {ir.ir_id: ir for ir in ir_list}
            if ir_id in ir_dict:
                ir = ir_dict[ir_id]
                # Find overlapping genes
                gene_mask = (
                    (genes["chrom"] == chrom)
                    & (genes["start"] < ir.end)
                    & (genes["end"] > ir.start)
                )
                gene_names = genes.loc[gene_mask, "name"].tolist()
                overlapping.append("; ".join(gene_names) if gene_names else "intergenic")
                continue

        overlapping.append("")

    features_df["overlapping_genes"] = overlapping
    return features_df


def _load_gff3_genes(gff3_path: Path) -> pd.DataFrame:
    """Parse gene features from a GFF3 annotation file.

    Args:
        gff3_path: Path to the GFF3 file.

    Returns:
        DataFrame with columns: chrom, start, end, name, feature_type.
    """
    rows = []
    try:
        with open(gff3_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] != "gene":
                    continue

                chrom = parts[0]
                # Normalise chromosome name
                if not chrom.startswith("Chr"):
                    chrom = f"Chr{chrom}"

                start = int(parts[3])
                end = int(parts[4])

                # Extract gene name from attributes
                attrs = parts[8]
                name = ""
                for attr in attrs.split(";"):
                    if attr.startswith("Name="):
                        name = attr.split("=", 1)[1]
                        break
                    elif attr.startswith("ID="):
                        name = attr.split("=", 1)[1]

                rows.append({
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "name": name,
                    "feature_type": "gene",
                })
    except Exception as e:
        logger.warning("Error parsing GFF3: %s", e)

    df = pd.DataFrame(rows)
    logger.info("Loaded %d gene annotations from GFF3", len(df))
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase3(cfg: Config) -> dict:
    """Execute Phase 3: clustering, prediction, and SHAP analysis.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dict with keys 'clusters', 'predictions', 'shap_features'.
    """
    out_dir = resolve_path(cfg, cfg.paths.phase3_output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load merged data from Phase 2
    merged_path = resolve_path(cfg, cfg.paths.phase2_output) / "merged_data.parquet"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"Phase 2 output not found at {merged_path}. Run Phase 2 first."
        )
    merged_df = pd.read_parquet(merged_path)
    logger.info("Loaded merged data: %d accessions, %d columns", *merged_df.shape)

    X, y = _split_features_targets(merged_df, cfg.params.target_traits)

    # Clustering
    logger.info("=== Clustering ===")
    umap_df = run_clustering(X, y, cfg, out_dir)

    # Prediction
    logger.info("=== Predictive Modelling ===")
    pred_results = run_prediction(X, y, cfg, out_dir)

    # SHAP
    logger.info("=== SHAP Analysis ===")
    gff3_path = resolve_path(cfg, cfg.paths.gff3_file)
    shap_results = run_shap_analysis(X, y, cfg, out_dir, gff3_path)

    logger.info("Phase 3 complete. Outputs saved to %s", out_dir)

    return {
        "clusters": umap_df,
        "predictions": pred_results,
        "shap_features": shap_results,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config()
    run_phase3(cfg)
