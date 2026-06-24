from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis


# =========================
# APP CONFIG
# =========================

app = Flask(__name__)

FS = 100

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# =========================
# LOAD PPG-ONLY MODEL
# =========================

model = joblib.load(MODELS_DIR / "glucose_model_ppg_only.pkl")
features_order = joblib.load(MODELS_DIR / "model_features_ppg_only.pkl")

print("PPG-ONLY MODEL LOADED SUCCESSFULLY", flush=True)
print("Number of model features:", len(features_order), flush=True)
print("Models dir:", MODELS_DIR, flush=True)


# =========================
# STABILITY SETTINGS
# =========================

USE_SMOOTHING = True

prediction_history = []

MAX_HISTORY = 3

# خليناه 30 بدل 40 عشان القفزات زي 172 تتفلتر أسرع
OUTLIER_THRESHOLD = 30.0

WARMUP_READINGS = 2


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API RUNNING",
        "model_type": "ppg_only_window_level_model_stable",
        "features": len(features_order),
        "smoothing": USE_SMOOTHING,
        "max_history": MAX_HISTORY,
        "outlier_threshold": OUTLIER_THRESHOLD,
        "metadata_used_by_model": False
    })


# =========================
# BASIC HELPERS
# =========================

def safe_div(a, b):
    return float(a / (b + 1e-6))


def clean_signal(x):
    x = np.array(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return x

    lower = np.percentile(x, 1)
    upper = np.percentile(x, 99)

    return np.clip(x, lower, upper)


def estimate_signal_bpm(x, fs=100, prominence_ratio=0.08):
    x = clean_signal(x)

    if len(x) < 500:
        return None, None, 0

    x = np.array(x, dtype=np.float64)

    signal_range = np.max(x) - np.min(x)
    signal_std = np.std(x)
    signal_mean = np.mean(x)

    cv = signal_std / (abs(signal_mean) + 1e-6)

    if signal_range < 50 or signal_std <= 0:
        return None, None, 0

    if cv < 0.0002:
        return None, None, 0

    min_distance = int(0.35 * fs)
    prominence = max(prominence_ratio * signal_std, 1e-6)

    peaks, _ = find_peaks(
        x,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 3 or len(peaks) > 18:
        return None, None, len(peaks)

    rr = np.diff(peaks) / fs

    rr = rr[
        (rr >= 0.35) &
        (rr <= 1.6)
    ]

    if len(rr) < 2:
        return None, None, len(peaks)

    bpm = 60 / (np.mean(rr) + 1e-6)

    if bpm < 40 or bpm > 170:
        return None, None, len(peaks)

    regularity = np.std(rr) / (np.mean(rr) + 1e-6)

    return float(bpm), float(regularity), len(peaks)


# =========================
# SIGNAL VALIDATION
# =========================

def is_valid_ppg_signal(ir, red):
    ir = np.array(ir, dtype=np.float64)
    red = np.array(red, dtype=np.float64)

    if len(ir) < 600 or len(red) < 600:
        return False, "Need at least 600 samples"

    if len(ir) != len(red):
        return False, "IR and RED length mismatch"

    if not np.all(np.isfinite(ir)) or not np.all(np.isfinite(red)):
        return False, "Signal contains invalid values"

    ir_mean = np.mean(ir)
    red_mean = np.mean(red)

    ir_clean = clean_signal(ir)
    red_clean = clean_signal(red)

    ir_std = np.std(ir_clean)
    red_std = np.std(red_clean)

    ir_range = np.max(ir_clean) - np.min(ir_clean)
    red_range = np.max(red_clean) - np.min(red_clean)

    ir_cv = ir_std / (abs(ir_mean) + 1e-6)
    red_cv = red_std / (abs(red_mean) + 1e-6)

    red_ir_ratio = red_mean / (ir_mean + 1e-6)

    corr = 0
    if ir_std > 0 and red_std > 0:
        corr = np.corrcoef(ir_clean, red_clean)[0, 1]

    ir_bpm, ir_regularity, ir_peaks = estimate_signal_bpm(
        ir,
        FS,
        prominence_ratio=0.08,
    )

    red_bpm, red_regularity, red_peaks = estimate_signal_bpm(
        red,
        FS,
        prominence_ratio=0.04,
    )

    print("SIGNAL DEBUG:", {
        "ir_mean": float(ir_mean),
        "red_mean": float(red_mean),
        "ir_std": float(ir_std),
        "red_std": float(red_std),
        "ir_range": float(ir_range),
        "red_range": float(red_range),
        "ir_cv": float(ir_cv),
        "red_cv": float(red_cv),
        "red_ir_ratio": float(red_ir_ratio),
        "corr": float(corr) if np.isfinite(corr) else None,
        "ir_bpm": ir_bpm,
        "red_bpm": red_bpm,
        "ir_regularity": ir_regularity,
        "red_regularity": red_regularity,
        "ir_peaks": ir_peaks,
        "red_peaks": red_peaks,
    }, flush=True)

    if ir_mean < 15000 or red_mean < 9000:
        return False, "Object is not in proper finger contact"

    if red_ir_ratio < 0.60 or red_ir_ratio > 1.05:
        return False, "Invalid RED/IR contact ratio"

    if not np.isfinite(corr):
        return False, "Invalid IR/RED correlation"

    # correlation أخف من النسخة اللي كانت بترفض كتير
    if corr < 0.45:
        return False, "IR/RED correlation too weak"

    # signal thresholds متوازنة
    if ir_range < 200 or red_range < 100:
        return False, "Signal range too weak for reliable prediction"

    if ir_cv < 0.0015 or red_cv < 0.001:
        return False, "PPG variation too weak for reliable prediction"

    if ir_std < 50 or red_std < 30:
        return False, "Signal amplitude too weak for reliable prediction"

    if ir_mean > 400000 or red_mean > 400000:
        return False, "Signal saturated"

    if ir_bpm is None:
        return False, "No valid IR pulse detected"

    if red_bpm is None:
        return False, "No valid RED pulse detected"

    if abs(ir_bpm - red_bpm) > 35:
        return False, "IR and RED pulse rates do not match"

    if ir_regularity is None or ir_regularity > 0.60:
        return False, "IR pulse is unstable"

    if red_regularity is None or red_regularity > 0.60:
        return False, "RED pulse is unstable"

    if ir_peaks < 3:
        return False, "Not enough IR pulse peaks"

    if red_peaks < 3:
        return False, "Not enough RED pulse peaks"

    if ir_peaks > 18:
        return False, "Too many random IR peaks"

    if red_peaks > 18:
        return False, "Too many random RED peaks"

    return True, "Valid finger PPG signal"


# =========================
# FEATURE EXTRACTION
# =========================

def basic_features(x):
    x = clean_signal(x)

    empty = {
        "mean": 0,
        "std": 0,
        "min": 0,
        "max": 0,
        "ptp": 0,
        "median": 0,
        "p25": 0,
        "p75": 0,
        "iqr": 0,
        "rms": 0,
        "energy": 0,
        "skew": 0,
        "kurtosis": 0,
        "ac_dc": 0,
        "cv": 0,
        "slope": 0,
        "slope_energy": 0,
        "mean_abs_diff": 0,
        "diff_mean": 0,
        "diff_std": 0,
        "zero_crossings": 0,
    }

    if len(x) == 0:
        return empty

    dx = np.diff(x)

    mean = np.mean(x)
    std = np.std(x)
    mn = np.min(x)
    mx = np.max(x)
    ptp = mx - mn

    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)

    if len(x) > 1:
        t = np.arange(len(x))
        slope = np.polyfit(t, x, 1)[0]
    else:
        slope = 0

    centered = x - mean

    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(mn),
        "max": float(mx),
        "ptp": float(ptp),
        "median": float(np.median(x)),
        "p25": float(p25),
        "p75": float(p75),
        "iqr": float(p75 - p25),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "energy": float(np.mean(x ** 2)),
        "skew": float(skew(x)) if len(x) > 2 and std > 0 else 0,
        "kurtosis": float(kurtosis(x)) if len(x) > 3 and std > 0 else 0,
        "ac_dc": safe_div(ptp, mean),
        "cv": safe_div(std, mean),
        "slope": float(slope),
        "slope_energy": float(np.mean(dx * dx)) if len(dx) > 0 else 0,
        "mean_abs_diff": float(np.mean(np.abs(dx))) if len(dx) > 0 else 0,
        "diff_mean": float(np.mean(dx)) if len(dx) > 0 else 0,
        "diff_std": float(np.std(dx)) if len(dx) > 0 else 0,
        "zero_crossings": float(
            np.sum(np.diff(np.sign(centered)) != 0)
        ) if len(centered) > 1 else 0,
    }


def peak_features(x, fs):
    x = clean_signal(x)

    empty = {
        "num_peaks": 0,
        "hr_bpm": 0,
        "rr_mean": 0,
        "rr_std": 0,
        "rr_min": 0,
        "rr_max": 0,
        "peak_amp_mean": 0,
        "peak_amp_std": 0,
        "peak_amp_min": 0,
        "peak_amp_max": 0,
        "pulse_regularity": 0,
        "peak_rate": 0,
    }

    if len(x) < 3:
        return empty

    min_distance = int(0.4 * fs)
    prom = max(0.05 * np.std(x), 1e-6)

    peaks, _ = find_peaks(
        x,
        distance=min_distance,
        prominence=prom,
    )

    if len(peaks) < 2:
        empty["num_peaks"] = float(len(peaks))
        empty["peak_rate"] = safe_div(len(peaks), len(x))
        return empty

    rr = np.diff(peaks) / fs
    peak_vals = x[peaks]

    return {
        "num_peaks": float(len(peaks)),
        "hr_bpm": float(60 / (np.mean(rr) + 1e-6)),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "rr_min": float(np.min(rr)),
        "rr_max": float(np.max(rr)),
        "peak_amp_mean": float(np.mean(peak_vals)),
        "peak_amp_std": float(np.std(peak_vals)),
        "peak_amp_min": float(np.min(peak_vals)),
        "peak_amp_max": float(np.max(peak_vals)),
        "pulse_regularity": float(np.std(rr) / (np.mean(rr) + 1e-6)),
        "peak_rate": safe_div(len(peaks), len(x)),
    }


def frequency_features(x, fs):
    x = clean_signal(x)

    empty = {
        "dom_freq": 0,
        "dom_power": 0,
        "band_power": 0,
        "band_power_ratio": 0,
        "low_band_power": 0,
        "high_band_power": 0,
        "low_high_power_ratio": 0,
        "spectral_entropy": 0,
    }

    if len(x) < 4:
        return empty

    x = x - np.mean(x)

    yf = np.abs(rfft(x))
    xf = rfftfreq(len(x), 1 / fs)

    yf = yf[1:]
    xf = xf[1:]

    if len(yf) == 0:
        return empty

    total_power = np.sum(yf ** 2) + 1e-6

    band_mask = (xf >= 0.5) & (xf <= 4.0)
    low_mask = (xf >= 0.5) & (xf < 1.5)
    high_mask = (xf >= 1.5) & (xf <= 4.0)

    if np.sum(band_mask) == 0:
        return empty

    xf_band = xf[band_mask]
    yf_band = yf[band_mask]

    power_band = yf_band ** 2
    band_power = np.sum(power_band)

    dom_idx = np.argmax(power_band)

    low_band_power = np.sum(yf[low_mask] ** 2) if np.sum(low_mask) > 0 else 0
    high_band_power = np.sum(yf[high_mask] ** 2) if np.sum(high_mask) > 0 else 0

    p = power_band / (band_power + 1e-6)

    return {
        "dom_freq": float(xf_band[dom_idx]),
        "dom_power": float(power_band[dom_idx]),
        "band_power": float(band_power),
        "band_power_ratio": float(band_power / total_power),
        "low_band_power": float(low_band_power),
        "high_band_power": float(high_band_power),
        "low_high_power_ratio": safe_div(low_band_power, high_band_power),
        "spectral_entropy": float(-np.sum(p * np.log2(p + 1e-12))),
    }


def relation_features(ir_w, red_w):
    ir_w = clean_signal(ir_w)
    red_w = clean_signal(red_w)

    min_len = min(len(ir_w), len(red_w))

    if min_len == 0:
        return {
            "ratio_mean_ir_red": 0,
            "ratio_std_ir_red": 0,
            "ratio_ptp_ir_red": 0,
            "ratio_acdc_ir_red": 0,
            "r_value": 0,
            "ir_red_corr": 0,
            "mean_diff_ir_red": 0,
            "std_diff_ir_red": 0,
            "motion_score": 0,
        }

    ir_w = ir_w[:min_len]
    red_w = red_w[:min_len]

    ir_mean = np.mean(ir_w)
    red_mean = np.mean(red_w)

    ir_std = np.std(ir_w)
    red_std = np.std(red_w)

    ir_ptp = np.ptp(ir_w)
    red_ptp = np.ptp(red_w)

    ir_acdc = safe_div(ir_ptp, ir_mean)
    red_acdc = safe_div(red_ptp, red_mean)

    if min_len > 2 and ir_std > 0 and red_std > 0:
        ir_red_corr = np.corrcoef(ir_w, red_w)[0, 1]
    else:
        ir_red_corr = 0

    ir_diff = np.diff(ir_w)
    red_diff = np.diff(red_w)

    motion_score = (
        np.mean(np.abs(ir_diff)) + np.mean(np.abs(red_diff))
    ) if len(ir_diff) > 0 and len(red_diff) > 0 else 0

    return {
        "ratio_mean_ir_red": safe_div(ir_mean, red_mean),
        "ratio_std_ir_red": safe_div(ir_std, red_std),
        "ratio_ptp_ir_red": safe_div(ir_ptp, red_ptp),
        "ratio_acdc_ir_red": safe_div(ir_acdc, red_acdc),
        "r_value": safe_div(
            safe_div(red_ptp, red_mean),
            safe_div(ir_ptp, ir_mean),
        ),
        "ir_red_corr": float(ir_red_corr),
        "mean_diff_ir_red": float(ir_mean - red_mean),
        "std_diff_ir_red": float(ir_std - red_std),
        "motion_score": float(motion_score),
    }


def extract_window_features(ir, red):
    ir = np.array(ir, dtype=np.float64)
    red = np.array(red, dtype=np.float64)

    ir = ir[np.isfinite(ir)]
    red = red[np.isfinite(red)]

    min_len = min(len(ir), len(red))

    if min_len == 0:
        raise ValueError("Empty IR or RED signal")

    ir = ir[:min_len]
    red = red[:min_len]

    f_ir_basic = basic_features(ir)
    f_red_basic = basic_features(red)

    f_ir_peak = peak_features(ir, FS)
    f_red_peak = peak_features(red, FS)

    f_ir_freq = frequency_features(ir, FS)
    f_red_freq = frequency_features(red, FS)

    f_relation = relation_features(ir, red)

    features = {
        **{f"ir_{k}": v for k, v in f_ir_basic.items()},
        **{f"red_{k}": v for k, v in f_red_basic.items()},
        **{f"ir_{k}": v for k, v in f_ir_peak.items()},
        **{f"red_{k}": v for k, v in f_red_peak.items()},
        **{f"ir_freq_{k}": v for k, v in f_ir_freq.items()},
        **{f"red_freq_{k}": v for k, v in f_red_freq.items()},
        **f_relation,
    }

    features = {
        k: 0 if not np.isfinite(v) else float(v)
        for k, v in features.items()
    }

    return features


# =========================
# OUTPUT HELPERS
# =========================

def classify_glucose_range(glucose):
    if glucose < 70:
        return "Low"
    elif glucose < 140:
        return "Normal"
    elif glucose < 180:
        return "Elevated"
    elif glucose < 250:
        return "High"
    else:
        return "Very High"


def get_confidence_and_warning(display_ready):
    if not display_ready:
        return (
            "warming_up",
            "Collecting stable PPG readings. Final value may change."
        )

    return (
        "medium",
        "Experimental non-invasive estimate. Not a replacement for glucometer."
    )


def stabilize_prediction(clipped_pred):
    global prediction_history

    outlier_rejected = False
    outlier_reason = None

    if not USE_SMOOTHING:
        prediction_history = [clipped_pred]
        return clipped_pred, outlier_rejected, outlier_reason

    if len(prediction_history) > 0:
        current_median = float(np.median(prediction_history))
        diff = abs(clipped_pred - current_median)

        if diff > OUTLIER_THRESHOLD:
            outlier_rejected = True
            outlier_reason = f"Reading jump {diff:.1f} mg/dL from recent median"

            final_pred = current_median

            print("OUTLIER REJECTED:", {
                "raw_clipped": clipped_pred,
                "recent_median": current_median,
                "diff": diff
            }, flush=True)

            return final_pred, outlier_rejected, outlier_reason

    prediction_history.append(clipped_pred)

    if len(prediction_history) > MAX_HISTORY:
        prediction_history.pop(0)

    final_pred = float(np.median(prediction_history))

    return final_pred, outlier_rejected, outlier_reason


# =========================
# PREDICTION ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if data is None:
            return jsonify({"error": "No JSON received"}), 400

        ir = data.get("ir")
        red = data.get("red")

        if ir is None or red is None:
            return jsonify({"error": "Missing ir or red"}), 400

        if len(ir) == 0 or len(red) == 0:
            return jsonify({"error": "Empty ir or red arrays"}), 400

        if len(ir) != len(red):
            return jsonify({"error": "ir and red length mismatch"}), 400

        # Metadata received only for logging.
        # It is NOT used by the PPG-only model.
        age = float(data.get("age", 25))
        gender = float(data.get("gender", 0))
        diabetic = int(data.get("diabetic", 0))
        fasting = int(data.get("fasting", 0))
        meal_time_hr = float(data.get("meal_time_hr", 2))
        motion_artifact = float(data.get("motion_artifact", 0))

        if fasting == 1:
            meal_time_hr = 0.0

        print(f"Received samples: IR={len(ir)}, RED={len(red)}", flush=True)

        print("INCOMING METADATA - NOT USED BY MODEL:", {
            "age": age,
            "gender": gender,
            "diabetic": diabetic,
            "fasting": fasting,
            "meal_time_hr": meal_time_hr,
            "motion_artifact": motion_artifact,
        }, flush=True)

        valid_signal, reason = is_valid_ppg_signal(ir, red)

        if not valid_signal:
            print("INVALID SIGNAL:", reason, flush=True)

            return jsonify({
                "error": "Invalid finger PPG signal",
                "reason": reason
            }), 422

        print("VALID SIGNAL:", reason, flush=True)

        features = extract_window_features(ir, red)

        X = pd.DataFrame([features])

        api_features = set(X.columns)
        model_features = set(features_order)

        matched = api_features.intersection(model_features)
        missing = model_features - api_features
        extra = api_features - model_features

        print("API FEATURES:", len(api_features), flush=True)
        print("MODEL FEATURES:", len(model_features), flush=True)
        print("MATCHED FEATURES:", len(matched), flush=True)
        print("MISSING FEATURES:", len(missing), flush=True)
        print("EXTRA FEATURES:", len(extra), flush=True)
        print("FIRST 20 MATCHED:", list(matched)[:20], flush=True)
        print("FIRST 20 MISSING:", list(missing)[:20], flush=True)

        X = X.reindex(columns=features_order, fill_value=0)

        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        raw_model_pred = float(model.predict(X)[0])

        clipped_pred = max(50, min(raw_model_pred, 450))

        final_pred, outlier_rejected, outlier_reason = stabilize_prediction(
            clipped_pred
        )

        display_ready = len(prediction_history) >= WARMUP_READINGS

        glucose_range = classify_glucose_range(final_pred)
        confidence, warning = get_confidence_and_warning(display_ready)

        print("MODEL USED: ppg_only_window_level_model_stable", flush=True)
        print("RAW MODEL PREDICTED:", raw_model_pred, flush=True)
        print("CLIPPED PREDICTED:", clipped_pred, flush=True)
        print("FINAL STABLE PREDICTED:", final_pred, flush=True)
        print("PREDICTION HISTORY:", prediction_history, flush=True)
        print("OUTLIER REJECTED:", outlier_rejected, outlier_reason, flush=True)
        print("DISPLAY READY:", display_ready, flush=True)
        print("RANGE:", glucose_range, flush=True)
        print("METADATA USED BY MODEL: False", flush=True)

        return jsonify({
            "predicted_glucose": round(float(final_pred), 1),
            "raw_glucose": round(float(raw_model_pred), 1),
            "clipped_glucose": round(float(clipped_pred), 1),
            "glucose_range": glucose_range,
            "confidence": confidence,
            "warning": warning,
            "model_used": "ppg_only_window_level_model_stable",
            "metadata_used_by_model": False,
            "smoothing_used": USE_SMOOTHING,
            "display_ready": display_ready,
            "warmup_readings": WARMUP_READINGS,
            "history_count": len(prediction_history),
            "prediction_history": [
                round(float(x), 1) for x in prediction_history
            ],
            "outlier_rejected": outlier_rejected,
            "outlier_reason": outlier_reason,
            "signal_status": reason,
            "used_features": len(features_order),
            "feature_debug": {
                "api_features": len(api_features),
                "model_features": len(model_features),
                "matched_features": len(matched),
                "missing_features": len(missing),
                "extra_features": len(extra),
                "first_20_matched": list(matched)[:20],
                "first_20_missing": list(missing)[:20],
            },
            "metadata_received": {
                "age": age,
                "gender": gender,
                "diabetic": diabetic,
                "fasting": fasting,
                "meal_time_hr": meal_time_hr,
                "motion_artifact": motion_artifact,
            }
        }), 200

    except Exception as e:
        print("ERROR:", e, flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST", "GET"])
def reset_prediction_history():
    global prediction_history

    prediction_history = []

    return jsonify({
        "status": "reset done",
        "prediction_history": prediction_history
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
    )
