#!/usr/bin/env python3
"""Generate embedding features for new webapp activity using saved UMAP models."""
import sys
import pickle
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.utils import check_random_state
from google import genai

from webapp_paths import ensure_src_paths
ensure_src_paths()

logger = logging.getLogger(__name__)

from generate_targets_embeddings import normalize_response_text
from compress_embeddings_umap import (
    l2_normalize,
    euclidean_distance,
    parse_country_location
)
from helpers_for_ratings_and_final_activity_features import pick_start_date
from get_codes_we_like import get_good_bad_and_target_codes, categorize_good_code, parse_dac_codes

# Paths
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data/trained_umap_models_trainval.pkl"

# Cache loaded models globally
_MODELS = None

_INT32_MIN = np.iinfo(np.int32).min + 1
_INT32_MAX = np.iinfo(np.int32).max - 1


def _find_ab_params(spread=1.0, min_dist=0.1):
    """Fit 1/(1+a*x^(2b)) to the UMAP piecewise target function."""
    x = np.linspace(0, spread * 3, 300)
    y_target = np.where(x <= min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    def _curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    params, _ = curve_fit(_curve, x, y_target)
    return float(params[0]), float(params[1])


def _sigma_binary_search(dists_knn, rho, target_k, n_iter=64):
    """Binary search for sigma such that 2^sum(p) = target_k."""
    target_sum = np.log2(float(target_k))
    lo, hi = 1e-5, 1e3
    sigma = 1.0
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        p = np.exp(-np.maximum(dists_knn - rho, 0.0) / mid)
        s = float(np.sum(p))
        if s < target_sum:
            lo = mid
        else:
            hi = mid
        sigma = mid
        if abs(s - target_sum) < 1e-5:
            break
    return sigma


def _tau_rand_int(state):
    """Tausworthe PRNG — exact copy of umap.utils.tau_rand_int (no numba)."""
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )
    return int(state[0] ^ state[1] ^ state[2])


def _umap_transform_single(x_new, raw_data, embedding, n_neighbors, a, b,
                            n_epochs=100, alpha=0.25, negative_sample_rate=5,
                            transform_seed=42):
    """
    UMAP transform for a single new point. Replicates umap-learn's
    optimize_layout_euclidean exactly:
      - epochs_per_sample stagger (max-weight edge fires every epoch)
      - edge thresholding: drop p < p_max / n_epochs
      - adaptive negative sample count per edge firing
      - Tausworthe PRNG seeded from transform_seed (default 42, same as umap-learn)
      - no p multiplier on gradients — edge frequency handles weighting

    No import of umap-learn or numba required.
    """
    N = raw_data.shape[0]

    # 1. k-NN in PCA space
    dists_all = np.sqrt(((x_new - raw_data) ** 2).sum(axis=1))
    knn_idx_part = np.argpartition(dists_all, n_neighbors)[:n_neighbors]
    order = np.argsort(dists_all[knn_idx_part])
    knn_idx = knn_idx_part[order]
    dists_knn = dists_all[knn_idx]

    # 2. rho, sigma, membership strengths
    rho = float(dists_knn[0])
    sigma = _sigma_binary_search(dists_knn, rho, n_neighbors)
    p = np.exp(-np.maximum(dists_knn - rho, 0.0) / sigma)

    # 3. Weighted-average initialisation
    w = p / (p.sum() + 1e-10)
    y = (w[:, None] * embedding[knn_idx]).sum(axis=0).copy().astype(np.float64)

    # 4. Edge thresholding (mirrors umap-learn graph.data thresholding)
    p_max = float(p.max())
    keep = p >= p_max / float(n_epochs)
    knn_idx = knn_idx[keep]
    p = p[keep]
    k_active = len(knn_idx)

    # 5. Epoch scheduling: max-weight edge fires every epoch
    eps = p_max / p                          # epochs_per_sample
    epoch_of_next_sample = eps.copy()        # first fire at epoch eps[i]
    eps_neg = eps / float(negative_sample_rate)
    epoch_of_next_neg = eps_neg.copy()

    # 6. Tausworthe RNG — matches umap-learn's transform_seed=42 default
    rs = check_random_state(transform_seed)
    rng_state = rs.randint(_INT32_MIN, _INT32_MAX, 3).astype(np.int64)

    # 7. SGD
    for n in range(n_epochs):
        lr = alpha * (1.0 - float(n) / float(n_epochs))

        for i in range(k_active):
            if epoch_of_next_sample[i] > n:
                continue

            # Attractive
            diff = y - embedding[knn_idx[i]]
            D2 = float(np.dot(diff, diff))
            if D2 > 0.0:
                D2b = D2 ** b
                coeff = 2.0 * a * b * (D2 ** (b - 1.0)) / (1.0 + a * D2b)
                y -= lr * np.clip(coeff * diff, -4.0, 4.0)

            epoch_of_next_sample[i] += eps[i]

            # Repulsive (adaptive count)
            n_neg = int((n - epoch_of_next_neg[i]) / eps_neg[i])
            for _ in range(n_neg):
                neg_k = abs(_tau_rand_int(rng_state)) % N
                diff_r = y - embedding[neg_k]
                D2_r = float(np.dot(diff_r, diff_r))
                if D2_r > 0.0:
                    D2b_r = max(D2_r, 1e-10) ** b
                    coeff_r = 2.0 * b / ((0.001 + D2_r) * (1.0 + a * D2b_r))
                    y += lr * np.clip(coeff_r * diff_r, -4.0, 4.0)
            if n_neg > 0:
                epoch_of_next_neg[i] += eps_neg[i] * n_neg

    return y


def load_models():
    """Load saved models and centroids (cached)."""
    global _MODELS
    if _MODELS is None:
        logger.info(f"Loading models from {MODEL_PATH}...")
        with MODEL_PATH.open('rb') as f:
            _MODELS = pickle.load(f)
        logger.info("Loaded: pca, umap3 arrays, centroids")

        # Pre-compute and cache a, b UMAP parameters (min_dist=0.1, spread=1.0)
        a, b = _find_ab_params(spread=1.0, min_dist=0.1)
        _MODELS['umap3_a'] = a
        _MODELS['umap3_b'] = b
        logger.info(f"UMAP curve params: a={a:.4f}, b={b:.4f}")
    return _MODELS


def pick_sector_for_activity(
    codes: set,
    good_codes: set,
    global_cat_counts: dict
) -> Optional[str]:
    """Reuse exact logic from compress_embeddings_umap.py (lines 244-263)."""

    good = codes & good_codes
    if not good:
        return None

    per_activity = Counter()
    for code in good:
        per_activity[categorize_good_code(code)] += 1

    if not per_activity:
        return None

    max_ct = max(per_activity.values())
    tied = [cat for cat, ct in per_activity.items() if ct == max_ct]
    if len(tied) == 1:
        return tied[0]

    # Tie-breaker: least common globally
    tied_sorted = sorted(tied, key=lambda c: (global_cat_counts.get(c, 0), c))
    return tied_sorted[0]

def process_new_activity(
    activity_text: str,
    dac_codes: set,  # e.g., {"23110", "23181"}
    recipient_iso3_fractions: str,  # "KEN:0.6|TZA:0.4"
    actual_start_date: Optional[str] = None,
    original_planned_start_date: Optional[str] = None,
    txn_first_date: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Process new activity and return embedding features.

    Returns:
        {
            'sector': str,
            'decade': int,
            'decade_label': str,
            'umap2_x', 'umap2_y',
            'umap3_x', 'umap3_y', 'umap3_z',
            'umap4_x', 'umap4_y', 'umap4_z', 'umap4_w',
            'sector_distance': float,
            'country_distance': float
        }
    """

    # Load models
    models = load_models()
    pca = models['pca']
    umap3_raw   = models['umap3_raw_data']
    umap3_emb   = models['umap3_embedding']
    umap3_k     = models['umap3_n_neighbors']
    sector_decade_centroid = models['sector_decade_centroid']
    country_decade_centroid = models['country_decade_centroid']
    decade_centroid = models['decade_centroid']
    global_cat_counts = models['global_cat_counts']
    GOOD_CODES = models['GOOD_CODES']

    # 1. Generate embedding (from J logic)
    cleaned_text = normalize_response_text(activity_text)

    # Use the google.genai.Client API (same as generate_targets_embeddings.py)
    from google import genai as genai_client
    from llm_tracing import wrap_genai_client
    client = wrap_genai_client(genai_client.Client(api_key=os.getenv("GEMINI_API_KEY")))

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=cleaned_text  # Note: 'contents' not 'content'
    )
    emb_raw = np.array(result.embeddings[0].values, dtype=np.float32)
    emb = l2_normalize(emb_raw)

    # 2. Determine decade FIRST (needed for sector inference)
    date_row = pd.Series({
        "actual_start_date": pd.to_datetime(actual_start_date, errors='coerce') if actual_start_date else pd.NaT,
        "original_planned_start_date": pd.to_datetime(original_planned_start_date, errors='coerce') if original_planned_start_date else pd.NaT,
        "txn_first_date": pd.to_datetime(txn_first_date, errors='coerce') if txn_first_date else pd.NaT,
    })

    start_dt = pick_start_date(date_row)
    if pd.isna(start_dt):
        decade = -1
        decade_label = "UNKNOWN"
    else:
        year = int(pd.Timestamp(start_dt).year)
        if year <= 2014:
            decade = 0
            decade_label = "le_2014"
        else:
            decade = 1
            decade_label = "ge_2015"

    # 3. Determine sector
    auto_detected = False
    inferred_from_embedding = False
    sector_candidates = None  # Will store top candidates when auto-detecting

    # Try to determine from DAC codes first
    if dac_codes:
        sector = pick_sector_for_activity(dac_codes, GOOD_CODES, global_cat_counts)
    else:
        sector = None

    # If no valid sector from DAC codes, infer from embedding
    if sector is None or sector == "Uncategorized":
        logger.warning("No valid DAC codes - inferring sector from embedding similarity...")

        # Find nearest sector-decade centroid
        same_decade_centroids = {
            (sec, dec): cent
            for (sec, dec), cent in sector_decade_centroid.items()
            if dec == decade
        }

        if not same_decade_centroids:
            raise ValueError(
                f"❌ No sector centroids available for decade={decade}\n"
                f"Cannot infer sector from embedding."
            )

        # Compute distance to each sector centroid
        distances = {}
        for (sec, _), cent in same_decade_centroids.items():
            distances[sec] = euclidean_distance(emb, cent)

        # Pick closest sector
        sector = min(distances, key=distances.get)
        auto_detected = True
        inferred_from_embedding = True

        # Store top 5 candidates with distances
        sector_candidates = sorted(distances.items(), key=lambda x: x[1])[:5]

        logger.info(f"Auto-detected sector: {sector} (distance: {distances[sector]:.3f})")
        logger.info(f"Other distances: {sector_candidates[:3]}")

    # Merge sectors (from K lines 360-363)
    if sector in ["Improving Energy Policy", "Clean Energy Generation"]:
        sector = "Energy"
    if sector == "Forestry & Sustainable Agriculture":
        sector = "General Environmental Protection"

    # 4. Parse countries (from K logic, lines 366-373)
    parsed = parse_country_location(recipient_iso3_fractions)
    if parsed is None:
        countries = ["GLOBAL"]
        weights = [1.0]
    else:
        countries = [c for c, _ in parsed]
        weights = [float(w) for _, w in parsed]

    # 5. Transform through PCA → real UMAP transform (no umap import needed)
    emb_2d = emb.reshape(1, -1)  # Shape (1, D)
    X_pca = pca.transform(emb_2d)  # (1, 50)

    # Ensemble over all Procrustes-aligned manifolds saved by compress_embeddings_umap.py.
    # Each entry in umap3_ensemble is a (N_train, 3) training embedding aligned to seed-0 via orthogonal
    # Procrustes. Calling _umap_transform_single on each and averaging gives a more stable position for
    # the new point than any single-seed transform.
    if 'umap3_ensemble' in models:
        coords_list = [
            _umap_transform_single(
                X_pca[0], umap3_raw, emb, umap3_k,
                a=models['umap3_a'], b=models['umap3_b']
            )
            for emb in models['umap3_ensemble']
        ]
        U3 = np.mean(coords_list, axis=0)
    else:
        U3 = _umap_transform_single(
            X_pca[0], umap3_raw, umap3_emb, umap3_k,
            a=models['umap3_a'], b=models['umap3_b']
        )  # (3,)

    # 6. Compute distances (from K logic, lines 518-533)
    s_key = (sector, decade)
    s_cent = sector_decade_centroid.get(s_key)

    if s_cent is None:
        raise ValueError(
            f"❌ No sector centroid found for sector='{sector}', decade={decade} ({decade_label})\n"
            f"Available sector-decade combinations in training data:\n"
            f"{sorted(sector_decade_centroid.keys())[:20]}...\n"
            f"This means the training data doesn't have enough examples of this sector in this time period.\n"
            f"Cannot compute reliable sector_distance without a reference centroid."
        )

    sector_distance = euclidean_distance(emb, s_cent)

    # Country distance (weighted mix, lines 525-533)
    if not countries or countries == ["GLOBAL"]:
        mix = decade_centroid[decade]
    else:
        mix = np.zeros_like(emb, dtype=np.float32)
        for c, w in zip(countries, weights):
            c_key = (c, decade)
            if c_key in country_decade_centroid:
                mix += float(w) * country_decade_centroid[c_key].astype(np.float32)
        if np.linalg.norm(mix) > 0:
            mix = l2_normalize(mix)
        else:
            # No valid countries, use decade fallback
            mix = decade_centroid[decade]

    country_distance = euclidean_distance(emb, mix)

    # 7. Save to output_dir if provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            'activity_id': output_dir.name,
            'sector': sector,
            'decade': int(decade),
            'decade_label': decade_label,
            'countries': countries,
            'country_weights': weights,
            'sector_distance': float(sector_distance),
            'country_distance': float(country_distance),
            'umap_3d': [float(U3[0]), float(U3[1]), float(U3[2])],
        }
        (output_dir / "targets_embedding_features.jsonl").write_text(json.dumps(record) + '\n')

    # 8. Return
    result = {
        'sector': sector,
        'sector_auto_detected': auto_detected,
        'inferred_from_embedding': inferred_from_embedding,
        'decade': int(decade),
        'decade_label': decade_label,
        'umap3_x': float(U3[0]),
        'umap3_y': float(U3[1]),
        'umap3_z': float(U3[2]),
        'sector_distance': float(sector_distance),
        'country_distance': float(country_distance),
    }

    # Add sector candidates if auto-detected
    if sector_candidates is not None:
        result['sector_candidates'] = [(sec, float(dist)) for sec, dist in sector_candidates]

    return result
