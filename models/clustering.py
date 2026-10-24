"""
Semantic Clustering and Dimensionality Reduction Module.
Groups student mistakes using Cosine Similarity, Agglomerative Hierarchical Clustering,
or DBSCAN, with PCA 2D projections and rigorous metric evaluations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


class MistakeClusterer:
    """
    Performs semantic grouping on student mistake vectors.
    Supports distance-threshold Agglomerative Clustering (recommended),
    DBSCAN, and KMeans, with graceful handling of small datasets and noise.
    """

    def __init__(
        self,
        default_method: str = "agglomerative",
        distance_threshold: float = 0.28,  # ~0.72 cosine similarity, optimal for all-MiniLM-L6-v2
        min_cluster_size: int = 2
    ):
        self.default_method = default_method
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    @staticmethod
    def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """Computes pairwise cosine similarity matrix S in [-1.0, 1.0]."""
        if len(embeddings) == 0:
            return np.empty((0, 0))
        return cosine_similarity(embeddings)

    def cluster_mistakes(
        self,
        embeddings: np.ndarray,
        records: List[Dict[str, Any]],
        method: Optional[str] = None,
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes clustering and returns comprehensive diagnostic report.
        """
        method = method or self.default_method
        dist_thresh = distance_threshold if distance_threshold is not None else self.distance_threshold
        n_samples = len(embeddings)

        if n_samples == 0:
            return {
                "n_samples": 0,
                "n_clusters": 0,
                "labels": [],
                "clusters": {},
                "outliers": [],
                "silhouette": None,
                "pca_coords": [],
                "pca_variance_ratio": [0.0, 0.0],
                "method_used": method,
                "similarity_matrix": []
            }

        sim_matrix = self.compute_similarity_matrix(embeddings)
        # Cosine distance: D = 1 - S, bounded to >= 0
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

        # 1. Clustering execution
        if n_samples == 1:
            labels = np.array([0])
        elif method == "kmeans":
            k = min(n_clusters or 3, n_samples)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        elif method == "dbscan":
            db = DBSCAN(eps=dist_thresh, min_samples=self.min_cluster_size, metric="precomputed")
            labels = db.fit_predict(dist_matrix)
        else:
            # Agglomerative clustering with distance threshold
            if n_clusters is not None:
                agg = AgglomerativeClustering(
                    n_clusters=min(n_clusters, n_samples),
                    metric="precomputed",
                    linkage="average"
                )
                labels = agg.fit_predict(dist_matrix)
            else:
                agg = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=dist_thresh,
                    metric="precomputed",
                    linkage="average"
                )
                labels = agg.fit_predict(dist_matrix)

        # 2. Outlier filtering for small isolated clusters in threshold mode
        unique_labels = sorted(set(labels))
        cluster_counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}

        final_labels = labels.copy()
        if method == "agglomerative" and n_clusters is None:
            # Reclassify isolated singletons whose max similarity to others is below threshold as outlier (-1)
            for lbl, count in cluster_counts.items():
                if count < self.min_cluster_size:
                    idx = np.where(labels == lbl)[0][0]
                    other_sims = [sim_matrix[idx, j] for j in range(n_samples) if j != idx]
                    if other_sims and max(other_sims) < (1.0 - dist_thresh):
                        final_labels[idx] = -1

        # 3. Silhouette evaluation
        silhouette = None
        sil_note = ""
        valid_cluster_labels = [l for l in final_labels if l != -1]
        unique_valid = set(valid_cluster_labels)
        if len(unique_valid) >= 2 and len(valid_cluster_labels) > len(unique_valid):
            try:
                # Compute on clustered (non-noise) points
                mask = final_labels != -1
                silhouette = float(silhouette_score(
                    dist_matrix[mask][:, mask],
                    final_labels[mask],
                    metric="precomputed"
                ))
            except Exception as e:
                sil_note = f"Silhouette calculation skipped: {str(e)}"
        else:
            sil_note = "Requires at least 2 distinct clusters with > 1 sample each to compute Silhouette score."

        # 4. PCA 2D Projection
        pca_coords, pca_var_ratio = self._compute_pca_projection(embeddings)

        # 5. Build Cluster Summaries
        clusters_summary = {}
        outliers_list = []

        for idx, rec in enumerate(records):
            lbl = int(final_labels[idx])
            rec_augmented = dict(rec)
            rec_augmented["cluster_label"] = lbl
            rec_augmented["pca_x"] = float(pca_coords[idx][0])
            rec_augmented["pca_y"] = float(pca_coords[idx][1])

            if lbl == -1:
                outliers_list.append(rec_augmented)
            else:
                clusters_summary.setdefault(lbl, []).append(rec_augmented)

        # Compute cluster centroids and representative medoids
        cluster_diagnostics = {}
        for lbl, members in clusters_summary.items():
            member_indices = [records.index(m) if m in records else i for i, m in enumerate(members)]
            # Find medoid (member with highest average similarity to other members)
            sub_sim = sim_matrix[np.ix_(member_indices, member_indices)]
            avg_sims = sub_sim.mean(axis=1) if len(member_indices) > 1 else np.array([1.0])
            best_idx = int(np.argmax(avg_sims))
            medoid = members[best_idx]

            # Common concept tags
            concepts = [m.get("concept_id") for m in members]
            dominant_concept = max(set(concepts), key=concepts.count) if concepts else "Unknown"

            # Ground truth tags if available
            gt_tags = [m.get("ground_truth_misconception") or m.get("misconception_tag") for m in members if m.get("misconception_tag") or m.get("ground_truth_misconception")]
            dominant_tag = max(set(gt_tags), key=gt_tags.count) if gt_tags else "Undetermined Pattern"

            cluster_diagnostics[lbl] = {
                "cluster_id": lbl,
                "size": len(members),
                "dominant_concept": dominant_concept,
                "dominant_misconception": dominant_tag,
                "representative_example": medoid,
                "mean_intra_similarity": float(np.mean(sub_sim)),
                "members": members
            }

        return {
            "n_samples": n_samples,
            "n_clusters": len(cluster_diagnostics),
            "labels": final_labels.tolist(),
            "cluster_diagnostics": cluster_diagnostics,
            "outliers": outliers_list,
            "silhouette": silhouette,
            "silhouette_note": sil_note,
            "pca_coords": pca_coords.tolist(),
            "pca_variance_ratio": pca_var_ratio,
            "method_used": method,
            "similarity_matrix": sim_matrix.tolist()
        }

    def _compute_pca_projection(self, embeddings: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Projects embeddings to 2D using PCA for visualization."""
        n_samples = len(embeddings)
        if n_samples == 0:
            return np.empty((0, 2)), [0.0, 0.0]
        if n_samples == 1:
            return np.zeros((1, 2)), [1.0, 0.0]

        n_comp = min(2, n_samples, embeddings.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        projected = pca.fit_transform(embeddings)

        if n_comp == 1:
            # Pad with 0 for Y axis
            projected = np.hstack([projected, np.zeros((n_samples, 1))])
            var_ratios = [float(pca.explained_variance_ratio_[0]), 0.0]
        else:
            var_ratios = [float(v) for v in pca.explained_variance_ratio_]

        return projected, var_ratios

    @staticmethod
    def compare_with_baseline_concept_tags(
        records: List[Dict[str, Any]],
        cluster_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Compares semantic clustering partition with simple concept-tag grouping.
        Reports Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
        against ground-truth misconception labels when available.
        """
        gt_labels = []
        concept_labels = []
        has_ground_truth = True

        for r in records:
            gt = r.get("ground_truth_misconception") or r.get("misconception_tag")
            if not gt or gt == "none_correct":
                has_ground_truth = False
            gt_labels.append(str(gt))
            concept_labels.append(str(r.get("concept_id", "unknown")))

        # Map strings to int codes
        concept_unique = {c: i for i, c in enumerate(set(concept_labels))}
        concept_codes = [concept_unique[c] for c in concept_labels]

        ari_semantic_vs_concept = float(adjusted_rand_score(concept_codes, cluster_labels))
        nmi_semantic_vs_concept = float(normalized_mutual_info_score(concept_codes, cluster_labels))

        results = {
            "ari_semantic_vs_concept": round(ari_semantic_vs_concept, 4),
            "nmi_semantic_vs_concept": round(nmi_semantic_vs_concept, 4),
            "num_concept_groups": len(set(concept_labels)),
            "num_semantic_clusters": len(set([l for l in cluster_labels if l != -1])),
            "has_ground_truth": has_ground_truth
        }

        if has_ground_truth:
            gt_unique = {g: i for i, g in enumerate(set(gt_labels))}
            gt_codes = [gt_unique[g] for g in gt_labels]
            
            ari_semantic_vs_gt = float(adjusted_rand_score(gt_codes, cluster_labels))
            ari_concept_vs_gt = float(adjusted_rand_score(gt_codes, concept_codes))
            nmi_semantic_vs_gt = float(normalized_mutual_info_score(gt_codes, cluster_labels))
            nmi_concept_vs_gt = float(normalized_mutual_info_score(gt_codes, concept_codes))

            results.update({
                "ari_semantic_vs_ground_truth": round(ari_semantic_vs_gt, 4),
                "ari_concept_vs_ground_truth": round(ari_concept_vs_gt, 4),
                "nmi_semantic_vs_ground_truth": round(nmi_semantic_vs_gt, 4),
                "nmi_concept_vs_ground_truth": round(nmi_concept_vs_gt, 4),
                "semantic_improvement_ari": round(ari_semantic_vs_gt - ari_concept_vs_gt, 4)
            })

        return results
