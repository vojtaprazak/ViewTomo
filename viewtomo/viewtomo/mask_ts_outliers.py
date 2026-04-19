#!/usr/bin/env python3
"""
automated masking for view tilt series
----------------------------------------------
Identifies and masks obscured (low-intensity) and vacuum (high-intensity) regions
using robust physics modeling (Beer-Lambert & Polynomial traces) across the tilt series.
"""

import sys
import os
import argparse
import mrcfile
import numpy as np
import scipy.ndimage as nd
import logging
import time
import psutil
import json

# Force headless backend for HPC compatibility (no X11 required)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# ==========================================
# MODULE 1: PHYSICS MODELING
# ==========================================

class PhysicsModel:
    """
    Handles peak detection and curve fitting across the tilt series to estimate 
    the intensity attenuation profiles of the vacuum and biological sample.
    """
    
    @staticmethod
    def bin_ndarray(data, binning=1):
        """Bins a 3D numpy array in the X and Y dimensions by the specified factor."""
        if binning <= 1: return data
        if data.ndim != 3:
            raise ValueError("PhysicsModel.bin_ndarray expects 3D input (Z,Y,X).")
            
        z, y, x = data.shape
        y_trunc = (y // binning) * binning
        x_trunc = (x // binning) * binning
        
        new_shape = (z, y_trunc//binning, x_trunc//binning)
        out = np.zeros(new_shape, dtype=np.float32)
        
        for i in range(z):
            view = data[i, :y_trunc, :x_trunc]
            view = view.reshape(y_trunc//binning, binning, x_trunc//binning, binning)
            out[i] = view.mean(axis=(1, 3))
            
        return out

    @staticmethod
    def get_histogram_stack(data, bins=512):
        sample = data.ravel()[::max(1, data.size // 100000)]
        d_min = np.min(sample)
        d_max = np.percentile(sample, 99.9)
        
        edges = np.linspace(d_min, d_max, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        
        hists = []
        for i in range(data.shape[0]):
            h, _ = np.histogram(data[i].ravel(), bins=edges)
            hists.append(h)
                
        return np.array(hists), centers

    @staticmethod
    def find_peaks_topographic(hist, centers):
        smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
        
        # Pass 1: Broad peaks (Sample, Vacuum)
        idx_broad, _ = find_peaks(smooth, prominence=np.max(smooth)*0.015, width=12)
        
        # Pass 2: Narrow peaks (Obscured rescue)
        idx_narrow, _ = find_peaks(smooth, prominence=np.max(smooth)*0.015, width=2)
        
        peaks = list(centers[idx_broad])
        
        range_min = centers[0]
        range_span = centers[-1] - centers[0]
        low_zone = range_min + 0.25 * range_span
        
        for idx in idx_narrow:
            val = centers[idx]
            if val < low_zone:
                is_distinct = True
                for existing in peaks:
                    if abs(val - existing) < (0.05 * range_span): 
                        is_distinct = False
                        break
                if is_distinct:
                    peaks.append(val)
        
        if len(peaks) == 0: 
            return np.array([])
        
        return np.sort(peaks)

    @staticmethod
    def estimate_tilt_offset(tilts, intensities, pretilt=None):
        mask = (~np.isnan(intensities)) & (intensities > 0)
        default_offset = pretilt if pretilt is not None else 0.0
        
        if np.sum(mask) < 5: 
            return default_offset
        
        t_train = tilts[mask]
        i_train = intensities[mask]
        
        # Huber with scaled polynomial features for extreme robustness
        model = make_pipeline(
            PolynomialFeatures(2, include_bias=False),
            StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=2000)
        )
        
        calculated_offset = default_offset
        try:
            model.fit(t_train.reshape(-1, 1), i_train)
            dense_t = np.linspace(-30, 30, 600)
            dense_y = model.predict(dense_t.reshape(-1, 1))
            
            if dense_y[300] > dense_y[0] and dense_y[300] > dense_y[-1]: 
                raw_calc = dense_t[np.argmax(dense_y)]
                if abs(raw_calc) < 25:
                    calculated_offset = raw_calc
        except:
            pass
            
        if pretilt is not None:
            if abs(calculated_offset - pretilt) > 10.0:
                return pretilt
            else:
                return calculated_offset
        
        return calculated_offset

    @staticmethod
    def fit_polynomial_trace(tilts, intensities, mask=None, degree=2):
        if mask is None: mask = np.ones(len(tilts), dtype=bool)
        
        valid = mask & (~np.isnan(intensities)) 
        t_train = tilts[valid]
        i_train = intensities[valid]
        
        if len(i_train) < 5: 
            return np.full_like(tilts, np.nan), -1.0, 1.0

        # Ridge penalty (alpha=0.1) explicitly penalizes wild diagonal slopes on flat data
        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            StandardScaler(),
            HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=2000)
        )
        
        try:
            model.fit(t_train.reshape(-1, 1), i_train)
            huber = model.named_steps['huberregressor']
            
            # Check for unphysical upward-curving valleys
            if degree == 2 and huber.coef_[1] > 0:
                degree = 1
                model = make_pipeline(
                    PolynomialFeatures(degree, include_bias=False),
                    StandardScaler(),
                    HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=2000)
                )
                model.fit(t_train.reshape(-1, 1), i_train)
                huber = model.named_steps['huberregressor']
            
            y_pred_valid = model.predict(t_train.reshape(-1, 1))
            
            mae = np.median(np.abs(i_train - y_pred_valid))
            rel_error = mae / (np.median(i_train) + 1e-6)
            
            if np.var(i_train) < 1e-4:
                score = 1.0 if np.std(i_train - y_pred_valid) < 1e-4 else 0.0
            else:
                score = r2_score(i_train, y_pred_valid)
            
            y_pred_full = model.predict(tilts.reshape(-1, 1))
            return y_pred_full, score, rel_error
        except:
            return np.full_like(tilts, np.nan), -1.0, 1.0

    @staticmethod
    def fit_beer_lambert(tilts, intensities, offset=0.0, mask=None, upper_bound_curve=None):
        if mask is None: mask = np.ones(len(tilts), dtype=bool)
        
        valid = mask & (~np.isnan(intensities)) & (intensities > 0)
        
        if upper_bound_curve is not None:
            constraint = intensities < (upper_bound_curve * 1.05)
            constraint |= np.isnan(upper_bound_curve)
            valid = valid & constraint

        t_train = tilts[valid]
        i_train = intensities[valid]
        
        if len(t_train) < 5: return np.full_like(tilts, np.nan), -1.0, 1.0

        rads = np.radians(t_train - offset)
        cos_val = np.cos(rads)
        valid_cos = cos_val > 0.05
        
        if np.sum(valid_cos) < 5: return np.full_like(tilts, np.nan), -1.0, 1.0
            
        X_phys = (1.0 / cos_val[valid_cos]).reshape(-1, 1)
        y_phys = np.log(i_train[valid_cos])
        
        model = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2000)
        
        try:
            model.fit(X_phys, y_phys)
            
            y_pred_valid_log = model.predict(X_phys)
            y_pred_valid_exp = np.exp(y_pred_valid_log)
            
            mae = np.median(np.abs(i_train - y_pred_valid_exp))
            rel_error = mae / (np.median(i_train) + 1e-6)
            
            score = r2_score(y_phys, y_pred_valid_log)
            
            rads_full = np.radians(tilts - offset)
            cos_full = np.cos(rads_full)
            valid_full = cos_full > 0.001
            
            y_pred_log = np.full_like(tilts, -np.inf)
            if np.any(valid_full):
                X_full = (1.0 / cos_full[valid_full]).reshape(-1, 1)
                y_pred_log[valid_full] = model.predict(X_full)
            
            return np.exp(y_pred_log), score, rel_error
        except:
            return np.full_like(tilts, np.nan), -1.0, 1.0

    @staticmethod
    def reclassify_peaks(tilts, all_raw_peaks, model_vac, model_samp, model_obs):
        new_vac, new_samp, new_obs = [], [], []

        for i in range(len(tilts)):
            peaks = all_raw_peaks[i]
            if len(peaks) == 0:
                new_vac.append(np.nan); new_samp.append(np.nan); new_obs.append(np.nan)
                continue

            v_pred = model_vac[i]
            s_pred = model_samp[i]
            o_pred = model_obs[i]

            if not np.isnan(v_pred):
                idx_v = np.argmin(np.abs(peaks - v_pred))
                val_v = peaks[idx_v]
                if abs(val_v - v_pred) < 0.3 * np.abs(v_pred):
                    new_vac.append(val_v)
                    peaks = np.delete(peaks, idx_v)
                else:
                    new_vac.append(np.nan)
            else:
                new_vac.append(np.nan)
            
            if len(peaks) == 0:
                new_samp.append(np.nan); new_obs.append(np.nan)
                continue

            if not np.isnan(s_pred):
                idx_s = np.argmin(np.abs(peaks - s_pred))
                val_s = peaks[idx_s]
                
                is_close = abs(val_s - s_pred) < 0.25 * np.abs(s_pred)
                is_below_vac = True
                if not np.isnan(v_pred):
                    if val_s > v_pred * 0.98: 
                        is_below_vac = False
                
                if is_close and is_below_vac:
                    new_samp.append(val_s)
                    peaks = np.delete(peaks, idx_s)
                else:
                    new_samp.append(np.nan)
            else:
                new_samp.append(np.nan)

            if len(peaks) == 0:
                new_obs.append(np.nan)
                continue

            if not np.isnan(o_pred):
                idx_o = np.argmin(np.abs(peaks - o_pred))
                val_o = peaks[idx_o]
                if abs(val_o - o_pred) < 0.5 * np.abs(o_pred) + 0.1 * np.ptp(peaks): 
                    new_obs.append(val_o)
                else:
                    new_obs.append(np.nan)
            else:
                new_obs.append(np.nan)

        return np.array(new_vac), np.array(new_samp), np.array(new_obs)

    @staticmethod
    def plot_debug(save_path, hists, extent, tilts, raw_traces, fitted_traces, fitted_models, scores_and_rels, thresholds, offset):
        vac_raw, samp_raw, obs_raw = raw_traces
        vac_fit_trace, samp_fit_trace, obs_fit_trace = fitted_traces
        vac_model, samp_model, obs_model = fitted_models
        
        vac_r2, samp_r2, obs_r2 = scores_and_rels[0]
        vac_rel, samp_rel, obs_rel = scores_and_rels[1]
        
        low_thresh, high_thresh = thresholds

        fig, ax = plt.subplots(figsize=(12, 9))
        
        im = ax.imshow(hists, aspect='auto', extent=extent, origin='lower', 
                       cmap='magma', interpolation='nearest', norm='log')
        
        ax.scatter(obs_fit_trace, tilts, c='cyan', s=15, alpha=0.6, marker='x', label='Fit Obscured')
        ax.scatter(samp_fit_trace, tilts, c='lime', s=15, alpha=0.6, marker='x', label='Fit Sample')
        ax.scatter(vac_fit_trace, tilts, c='yellow', s=15, alpha=0.6, marker='x', label='Fit Vacuum')

        ax.plot(samp_model, tilts, color='white', linestyle='-', lw=2, label=f'Sample Fit (R²={samp_r2:.2f}, Err={samp_rel*100:.1f}%)')
        ax.plot(obs_model, tilts, color='cyan', linestyle='--', lw=1.5, label=f'Obsc Fit (R²={obs_r2:.2f}, Err={obs_rel*100:.1f}%)')
        ax.plot(vac_model, tilts, color='yellow', linestyle='--', lw=1.5, label=f'Vac Fit (R²={vac_r2:.2f}, Err={vac_rel*100:.1f}%)')

        if low_thresh[0] is not None and not np.isnan(low_thresh[0]):
            ax.plot(low_thresh, tilts, color='magenta', linestyle=':', lw=2, label='Lower Cut')
        if high_thresh[0] is not None and not np.isnan(high_thresh[0]):
            ax.plot(high_thresh, tilts, color='orange', linestyle=':', lw=2, label='Upper Cut')
        
        ax.axhline(offset, color='red', linestyle='--', alpha=0.5, label=f'Zero-Tilt ({offset:.1f}°)')

        ax.set_title(f"Physics Fit Debug\nOffset: {offset:.1f}deg | Fit Range: +/-40deg")
        ax.set_ylabel("Tilt Angle (deg)")
        ax.set_xlabel("Intensity")
        ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
        
        plt.savefig(save_path, dpi=150)
        plt.close()

    @classmethod
    def calculate_thresholds(cls, logger, data, n_tilts, cut_factor_low=0.25, cut_factor_high=0.25, wiggle=1.0, binning=16, pretilt=None, debug=False, debug_path="debug_fit.png"):
        logger.info(f"Binning data by factor {binning} for histogram analysis...")
        binned_for_stats = cls.bin_ndarray(data, binning=binning)
        
        tilts = np.linspace(-60, 60, n_tilts)
        hists, centers = cls.get_histogram_stack(binned_for_stats)
        
        vac_trace, samp_trace, obs_trace = [], [], []
        all_raw_peaks = []
        range_span = centers[-1] - centers[0]
        
        for h in hists:
            peaks = cls.find_peaks_topographic(h, centers)
            all_raw_peaks.append(peaks)
            v, s, o = np.nan, np.nan, np.nan
            
            if len(peaks) > 0:
                if peaks[0] < (centers[0] + 0.20 * range_span):
                    o = peaks[0]
                    peaks = peaks[1:]
                if len(peaks) > 0 and peaks[-1] > (centers[-1] - 0.3 * range_span):
                    v = peaks[-1]
                    peaks = peaks[:-1]
                if len(peaks) > 0:
                    s = peaks[-1] 
                    
            vac_trace.append(v); samp_trace.append(s); obs_trace.append(o)

        vac_trace = np.array(vac_trace); samp_trace = np.array(samp_trace); obs_trace = np.array(obs_trace)
        
        final_offset = cls.estimate_tilt_offset(tilts, samp_trace, pretilt=pretilt)
        logger.info(f"Final Tilt Offset: {final_offset:.1f} degrees")
        trust_mask = np.abs(tilts - final_offset) <= 40.0
        
        vac_m1, _, _ = cls.fit_polynomial_trace(tilts, vac_trace, mask=trust_mask)
        samp_m1, _, _ = cls.fit_beer_lambert(tilts, samp_trace, offset=final_offset, mask=trust_mask, upper_bound_curve=vac_m1)
        obs_m1, _, _ = cls.fit_beer_lambert(tilts, obs_trace, offset=final_offset, mask=trust_mask, upper_bound_curve=samp_m1)
        
        logger.debug("Reclassifying peaks based on initial fit...")
        vac_clean, samp_clean, obs_clean = cls.reclassify_peaks(tilts, all_raw_peaks, vac_m1, samp_m1, obs_m1)
        
        vac_final, vac_score, vac_rel = cls.fit_polynomial_trace(tilts, vac_clean, mask=trust_mask)
        samp_final, samp_score, samp_rel = cls.fit_beer_lambert(tilts, samp_clean, offset=final_offset, mask=trust_mask, upper_bound_curve=vac_final)
        obs_final, obs_score, obs_rel = cls.fit_beer_lambert(tilts, obs_clean, offset=final_offset, mask=trust_mask, upper_bound_curve=samp_final)
        
        logger.info(f"Vacuum Fit R²:   {vac_score:.2f} (Rel Error: {vac_rel*100:.1f}%)")
        logger.info(f"Sample Fit R²:   {samp_score:.2f} (Rel Error: {samp_rel*100:.1f}%)")
        logger.info(f"Obscured Fit R²: {obs_score:.2f} (Rel Error: {obs_rel*100:.1f}%)")

        low_thresholds = np.full_like(tilts, np.nan)
        high_thresholds = np.full_like(tilts, np.nan)
        
        vac_good = (vac_score > 0.4) or (vac_rel < 0.15)
        samp_good = (samp_score > 0.4) or (samp_rel < 0.15)
        obs_good = (obs_score > 0.4) or (obs_rel < 0.15)
        
        success = False
        base_cut_low = cut_factor_low
        base_cut_high = cut_factor_high

        if samp_good:
            success = True
            eff_cut_low = base_cut_low

            if obs_good:
                center_mask = np.abs(tilts - final_offset) < 10
                edge_mask = np.abs(tilts - final_offset) > 40
                
                gap_center = np.median(samp_final[center_mask] - obs_final[center_mask])
                if np.sum(edge_mask) > 0:
                    gap_edge = np.median(samp_final[edge_mask] - obs_final[edge_mask])
                    if gap_center > 0 and (gap_edge / gap_center) > 0.95:
                        logger.warning(f">> Robustness Alert: Sample trace does not decay relative to Obscured.")
                        if samp_score < 0.8:
                            success = False
            
            if success:
                if obs_good:
                    low_thresholds = obs_final + (samp_final - obs_final) * eff_cut_low
                else:
                    low_thresholds = samp_final * eff_cut_low

                # Guarantees upper cut decays exactly like the biological sample, solving the inverted curve issue.
                if vac_good:
                    idx_zero = np.argmin(np.abs(tilts - final_offset))
                    anchor_val = vac_final[idx_zero] - (vac_final[idx_zero] - samp_final[idx_zero]) * base_cut_high
                    attenuation_profile = samp_final / samp_final[idx_zero]
                    high_thresholds = anchor_val * attenuation_profile
                else:
                    logger.info("   [Fallback] Vacuum fit poor. Using sample-relative upper threshold.")
                    high_thresholds = samp_final * (1.0 + abs(base_cut_high))

                # Apply secant-based relaxation via wiggle factor at high tilts
                rads = np.radians(tilts - final_offset)
                secant = 1.0 / np.clip(np.cos(rads), 0.1, 1.0)
                tilt_relax = 1.0 + (secant - 1.0) * wiggle
                
                low_thresholds = low_thresholds / tilt_relax
                high_thresholds = high_thresholds * tilt_relax

        if debug:
            extent = [centers[0], centers[-1], np.min(tilts), np.max(tilts)]
            cls.plot_debug(debug_path, hists, extent, tilts, 
                           (vac_trace, samp_trace, obs_trace), 
                           (vac_clean, samp_clean, obs_clean),
                           (vac_final, samp_final, obs_final),
                           ((vac_score, samp_score, obs_score), (vac_rel, samp_rel, obs_rel)),
                           (low_thresholds, high_thresholds),
                           final_offset)
                           
            json_path = debug_path.replace('_debug.png', '_params.json')
            params_dump = {
                "tilt_offset_deg": float(final_offset),
                "success": bool(success),
                "metrics": {
                    "vacuum": {"r2": float(vac_score), "rel_error_pct": float(vac_rel * 100)},
                    "sample": {"r2": float(samp_score), "rel_error_pct": float(samp_rel * 100)},
                    "obscured": {"r2": float(obs_score), "rel_error_pct": float(obs_rel * 100)}
                },
                "raw_traces": {
                    "tilts": tilts.tolist(),
                    "vacuum": [None if np.isnan(x) else float(x) for x in vac_trace],
                    "sample": [None if np.isnan(x) else float(x) for x in samp_trace],
                    "obscured": [None if np.isnan(x) else float(x) for x in obs_trace]
                },
                "cleaned_traces": {
                    "vacuum": [None if np.isnan(x) else float(x) for x in vac_clean],
                    "sample": [None if np.isnan(x) else float(x) for x in samp_clean],
                    "obscured": [None if np.isnan(x) else float(x) for x in obs_clean]
                },
                "fitted_models": {
                    "vacuum": [None if np.isnan(x) else float(x) for x in vac_final],
                    "sample": [None if np.isnan(x) else float(x) for x in samp_final],
                    "obscured": [None if np.isnan(x) else float(x) for x in obs_final]
                },
                "thresholds": {
                    "low_cut": [None if np.isnan(x) else float(x) for x in low_thresholds],
                    "high_cut": [None if np.isnan(x) else float(x) for x in high_thresholds],
                }
            }
            try:
                with open(json_path, 'w') as f:
                    json.dump(params_dump, f, indent=4)
                logger.info(f"Saved extended machine-readable fit parameters to {json_path}")
            except Exception as e:
                logger.error(f"Failed to write JSON debug params: {e}")

        return low_thresholds, high_thresholds, success

# ==========================================
# MODULE 2: IMAGE PROCESSING
# ==========================================

class ImageProcessor:
    """
    Handles the 2D image processing tasks for individual projection slices,
    including robust thresholding, edge detection, morphology, and inpainting.
    """
    @staticmethod
    def robust_stats(image, center_fraction=0.25):
        """Calculates median and IQR from the central region of an image."""
        y, x = image.shape
        cy, cx = y // 2, x // 2
        dy, dx = max(int(y * center_fraction), 1), max(int(x * center_fraction), 1)
        crop = image[cy-dy:cy+dy, cx-dx:cx+dx]
        return np.median(crop), np.subtract(*np.percentile(crop, [75, 25]))

    @staticmethod
    def generate_mask(image, low_cut, high_cut, med, iqr, sigma=2.0, dilation=6, dust_threshold=20000):
        if low_cut is not None and not np.isnan(low_cut):
            t_min = low_cut
        else:
            t_min = max(med - 3 * iqr, med * 0.15) 

        if high_cut is not None and not np.isnan(high_cut):
            t_max = high_cut
        else:
            t_max = med + 3.5 * iqr 

        mask_thresh = (image < t_min) | (image > t_max)
        blurred = nd.gaussian_filter(image, sigma=sigma)
        grad = np.hypot(nd.sobel(blurred, axis=0), nd.sobel(blurred, axis=1))
        
        g_med, g_iqr = ImageProcessor.robust_stats(grad)
        mask_edge = grad > (g_med + 3.0 * g_iqr)

        combined = mask_thresh | mask_edge
        struct = nd.generate_binary_structure(2, 1) 
        
        # Pad before morphology to prevent the border from eroding inwards 
        pad_sz = max(5, dilation, 2) + 1
        combined_padded = np.pad(combined, pad_width=pad_sz, mode='edge')
        
        processed = nd.binary_closing(combined_padded, structure=struct, iterations=5)
        processed = nd.binary_dilation(processed, structure=struct, iterations=dilation)
        processed = nd.binary_opening(processed, structure=struct, iterations=2)
        
        # Crop back down to original dimensions
        processed = processed[pad_sz:-pad_sz, pad_sz:-pad_sz]

        if dust_threshold > 0:
            eff_dust = dust_threshold + (2 * dilation * (dilation + 1))
            labeled_mask, num_labels = nd.label(processed, structure=struct)
            if num_labels > 0:
                label_counts = np.bincount(labeled_mask.ravel())
                mask_sizes = label_counts > eff_dust
                mask_sizes[0] = 0
                processed = mask_sizes[labeled_mask]
        
        # Border clearing has been completely removed so physical anomalies at edges are kept masked
        return processed

    @staticmethod
    def inpaint_slice(image, mask, soft_sigma=10.0):
        valid_pixels = image[~mask]
        if valid_pixels.size < 100: 
            safe_mean, safe_std = np.mean(image), np.std(image)
        else:
            safe_mean, safe_std = np.mean(valid_pixels), np.std(valid_pixels)
        
        noise = np.random.normal(safe_mean, safe_std, image.shape)
        noise = nd.gaussian_filter(noise, sigma=1.0) 
        
        # Enforce 'nearest' edge mode to prevent blending dropoff right at the frame borders
        alpha = nd.gaussian_filter(mask.astype(float), sigma=soft_sigma, mode='nearest')
        alpha = np.clip(alpha, 0, 1)

        output = (1.0 - alpha) * image + alpha * noise
        return output.astype(image.dtype)

# ==========================================
# MODULE 3: WORKERS & MAIN APP
# ==========================================

def _worker_generate_mask(args):
    slice_data, low, high, dilation, dust, sigma = args
    med, iqr = ImageProcessor.robust_stats(slice_data)
    return ImageProcessor.generate_mask(slice_data, low, high, med, iqr, sigma, dilation, dust)

def _worker_inpaint(args):
    i, data_slice, mask_slice, binning, dilation, softness = args
    h, w = data_slice.shape
    
    m_full = mask_slice.repeat(binning, axis=0).repeat(binning, axis=1)
    m_full = m_full[:h, :w]
    if m_full.shape != (h, w):
        pad_h = max(0, h - m_full.shape[0])
        pad_w = max(0, w - m_full.shape[1])
        # Use mode='edge' so scale differences near boundaries aren't padded with False (0)
        m_full = np.pad(m_full, ((0, pad_h), (0, pad_w)), mode='edge')
    
    return i, ImageProcessor.inpaint_slice(data_slice, m_full, soft_sigma=softness)

class AutoMasker:
    """
    Orchestrator class for the automated masking pipeline.
    Coordinates physics-based parameter estimation, parallel mask generation, 
    and parallel noise-matched inpainting.
    """
    def __init__(self, args):
        self.args = args
        self.saved = False
        self.sigma = 2
        
        base_dir = os.path.dirname(args.output)
        base_name = os.path.basename(args.output).replace('_masked.mrc', '')
        self.log_file = os.path.join(base_dir, f"{base_name}_masking.log")
        self.debug_png = os.path.join(base_dir, f"{base_name}_masking_debug.png")
        
        self.logger = self.setup_logger(self.log_file, args.debug)
        self.logger.info("="*50)
        self.logger.info(f"Starting AutoMasker for {base_name}")
        self.logger.info("="*50)

        self.logger.info(f"Loading {args.input}...")     
        t0 = time.time()
        with mrcfile.mmap(args.input, mode='r') as mrc:
            self.data = mrc.data
            self.voxel_size = mrc.voxel_size
            self.header_template = mrc.header.copy()
            self.ext_header = mrc.extended_header.copy() if mrc.extended_header is not None else None

        self.binned_data = self.data[:, ::args.binning, ::args.binning]
        self.masks = np.zeros_like(self.binned_data, dtype=bool)
        
        self.logger.info(f"Data loaded in {time.time()-t0:.2f}s. Shape: {self.data.shape}, Binned: {self.binned_data.shape}")
        
        h_factor = args.high_cut_factor if getattr(args, 'high_cut_factor', None) is not None else getattr(args, 'cut_factor', 0.25)
        
        self.low_thresholds, self.high_thresholds, success = PhysicsModel.calculate_thresholds(
            self.logger,
            self.data, 
            n_tilts=self.data.shape[0],
            cut_factor_low=getattr(args, 'cut_factor', 0.25),
            cut_factor_high=h_factor,
            wiggle=getattr(args, 'wiggle', 1.0),
            binning=getattr(args, 'hist_binning', 16),
            pretilt=getattr(args, 'pretilt', None),
            debug=args.debug,
            debug_path=self.debug_png
        )
        
        if not success:
            self.logger.warning("Physics fit unreliable. Reverting to robust statistical thresholding.")

        dust_area_thresh = max(1, int(args.dust / (args.binning**2)))
        workers = self.get_dynamic_workers()
        self.logger.info(f"Generating masks ({workers} workers)...")

        t1 = time.time()
        map_args = []
        for i in range(self.binned_data.shape[0]):
            l_cut = self.low_thresholds[i] if isinstance(self.low_thresholds, (np.ndarray, list)) else None
            h_cut = self.high_thresholds[i] if isinstance(self.high_thresholds, (np.ndarray, list)) else None
            map_args.append((self.binned_data[i], l_cut, h_cut, args.dilation, dust_area_thresh, self.sigma))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_worker_generate_mask, map_args, chunksize=5)
            self.masks = np.array(list(results))
        self.logger.info(f"Mask generation completed in {time.time()-t1:.2f}s.")

        self.save_and_exit()

    def setup_logger(self, log_path, debug_mode):
        logger = logging.getLogger("AutoMasker")
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
        
        return logger

    def get_dynamic_workers(self):
        """
        Calculates optimal worker count based on CPU limits, system load, and memory.
        Respects SLURM limits if running on an HPC cluster.
        """
        # 1. Check for SLURM allocation first to prevent node throttling
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus is not None:
            cores = int(slurm_cpus)
            workers = max(1, cores - 1)  # Leave 1 core for the main thread
            self.logger.debug(f"Detected SLURM allocation: {cores} cores.")
        else:
            # 2. Fall back to physical cores and load average
            cores = psutil.cpu_count(logical=False) or os.cpu_count()
            load_avg = os.getloadavg()[0] 
            available_cores = max(1, cores - load_avg)
            workers = max(1, int(available_cores * 0.7))
        
        # 3. Memory limits (prevents OOM crashes on enormous binned slices)
        available_mem_gb = psutil.virtual_memory().available / (1024**3)
        slice_size_gb = self.binned_data[0].nbytes / (1024**3)
        mem_limited_workers = max(1, int(available_mem_gb / (max(slice_size_gb, 0.001) * 5))) 
        
        final_workers = min(workers, mem_limited_workers, getattr(self.args, 'workers', 8))
        
        self.logger.debug(f"System Check -> Cores: {cores}, Avail Mem: {available_mem_gb:.1f}GB")
        self.logger.debug(f"Calculated limits -> CPU-based: {workers}, Memory-based: {mem_limited_workers}")
        
        return final_workers

    def save_and_exit(self):
        mask_out = self.args.output.replace('.mrc', '_mask.mrc')
        self.logger.info(f"Saving binary mask to {mask_out}...")
        with mrcfile.new(mask_out, overwrite=True) as mrc:
            mrc.set_data(self.masks.astype(np.int16))
            vs = self.voxel_size
            mrc.voxel_size = (vs.x * self.args.binning, vs.y * self.args.binning, vs.z)
            if self.ext_header is not None: mrc.set_extended_header(self.ext_header)

        if not self.args.trial:
            num_slices = self.data.shape[0]
            self.logger.info(f"Homogenizing and saving full-res stack to {self.args.output}...")
            
            out_data = np.zeros_like(self.data)
            workers = self.get_dynamic_workers()
            
            t2 = time.time()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_worker_inpaint, (i, self.data[i], self.masks[i], self.args.binning, getattr(self.args, 'dilation', 6), getattr(self.args, 'softness', 30.0))): i 
                    for i in range(num_slices)
                }
                completed = 0
                for future in as_completed(futures):
                    idx, processed_slice = future.result() 
                    out_data[idx] = processed_slice
                    completed += 1
                    sys.stdout.write(f"\r  Progress: {completed}/{num_slices} slices ({(completed/num_slices)*100:.1f}%)")
                    sys.stdout.flush()
            
            print("\nWriting MMap to disk...")
            with mrcfile.new(self.args.output, overwrite=True) as mrc:
                mrc.set_data(out_data)
                for f in self.header_template.dtype.names: mrc.header[f] = self.header_template[f]
                mrc.voxel_size = self.voxel_size
                if self.ext_header is not None: mrc.set_extended_header(self.ext_header)
                
            self.logger.info(f"Inpainting completed in {time.time()-t2:.2f}s.")
            
        self.saved = True
        self.logger.info("AutoMasker finished successfully.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-based outlier masking and inpainting.")
    parser.add_argument("input", help="Input MRC stack")
    parser.add_argument("output", help="Output homogenized filename")
    parser.add_argument("--binning", type=int, default=6, help="Binning factor for Mask Generation")
    parser.add_argument("--hist_binning", type=int, default=16, help="Binning factor for Histogram Stats. Default 16")
    parser.add_argument("--trial", action='store_true', help='Save mask only', default=False)
    parser.add_argument("--dilation", type=int, default=6, help="Mask dilation iterations. Default 6.")
    parser.add_argument("--softness", type=float, default=30.0, help="Soft edge gradient length (full-res px)")
    
    # Threshold Controls
    parser.add_argument("--cut_factor", type=float, default=0.05, help="Lower (Obscured) threshold factor (0.0-1.0)")
    parser.add_argument("--high_cut_factor", type=float, default=0.05, help="Upper (Vacuum) threshold factor. Defaults to cut_factor if None.")
    
    parser.add_argument("--wiggle", type=float, default=1.0, help="Relaxation factor. Widen margins geometrically at high tilts.")
    parser.add_argument("--pretilt", type=float, default=None, help="Expected zero-tilt angle (degrees) to guide offset fitting")
    parser.add_argument("--dust", type=int, default=20000, help="Dust exclusion threshold.")
    parser.add_argument("--debug", action='store_true', help="Save plot of physics fits and log info")
    parser.add_argument("--workers", type=int, default=8, help="Max CPU workers to use")
    
    app = AutoMasker(parser.parse_args())
    if not app.saved: sys.exit(1)