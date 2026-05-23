from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

app = Flask(__name__)

FS = 100

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "glucose_model.pkl")
features_order = joblib.load(BASE_DIR / "models" / "model_features.pkl")

print("MODEL LOADED SUCCESSFULLY")
print("Number of model features:", len(features_order))


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API RUNNING"
    })


def safe_div(a, b):
    return float(a / (b + 1e-6))


def clean_signal(x):
    x = np.array(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return x

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr

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

    ir_std = np.std(ir)
    red_std = np.std(red)

    ir_range = np.max(ir) - np.min(ir)
    red_range = np.max(red) - np.min(red)

    ir_cv = ir_std / (abs(ir_mean) + 1e-6)
    red_cv = red_std / (abs(red_mean) + 1e-6)

    corr = 0
    if ir_std > 0 and red_std > 0:
        corr = np.corrcoef(ir, red)[0, 1]

    # IR is the main pulse channel.
    ir_bpm, ir_regularity, ir_peaks = estimate_signal_bpm(
        ir,
        FS,
        prominence_ratio=0.08,
    )

    # RED is supportive, not mandatory.
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
        "corr": float(corr) if np.isfinite(corr) else None,
        "ir_bpm": ir_bpm,
        "red_bpm": red_bpm,
        "ir_regularity": ir_regularity,
        "red_regularity": red_regularity,
        "ir_peaks": ir_peaks,
        "red_peaks": red_peaks,
    })

    if ir_mean < 3000 or red_mean < 300:
        return False, "Weak finger contact"

    if ir_mean > 400000 or red_mean > 400000:
        return False, "Signal saturated"

    if ir_range < 80 or red_range < 30:
        return False, "Signal is too flat"

    if ir_cv < 0.00025 or red_cv < 0.00008:
        return False, "PPG variation too weak"

    if not np.isfinite(corr):
        return False, "Invalid IR/RED correlation"

    if corr < 0.45:
        return False, "IR/RED signals are not correlated"

    # Main requirement: valid IR pulse.
    if ir_bpm is None:
        return False, "No valid IR pulse detected"

    if ir_regularity is None or ir_regularity > 0.70:
        return False, "IR pulse is not regular enough"

    if ir_peaks < 3:
        return False, "Not enough IR pulse peaks"

    if ir_peaks > 18:
        return False, "Too many random IR peaks"

    # RED check:
    # If RED bpm exists, compare it with IR.
    # If RED bpm does not exist, accept only when IR/RED correlation is high.
    if red_bpm is not None:
        if abs(ir_bpm - red_bpm) > 35:
            return False, "IR and RED pulse rates do not match"

        if red_regularity is not None and red_regularity > 0.75:
            return False, "RED pulse is not regular enough"
    else:
        if corr < 0.85:
            return False, "RED pulse not detected and correlation is not high enough"

        if red_peaks < 1:
            return False, "No RED pulse activity detected"

    return True, "Valid finger PPG signal"


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


def extract_features(ir, red):
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

        print(f"Received samples: IR={len(ir)}, RED={len(red)}")

        valid_signal, reason = is_valid_ppg_signal(ir, red)

        if not valid_signal:
            print("INVALID SIGNAL:", reason)
            return jsonify({
                "error": "Invalid finger PPG signal",
                "reason": reason
            }), 422

        print("VALID SIGNAL:", reason)

        features = extract_features(ir, red)

        features["age"] = data.get("age", 25)
        features["gender"] = data.get("gender", 0)
        features["diabetic"] = data.get("diabetic", 0)
        features["fasting"] = data.get("fasting", 0)
        features["meal_time_hr"] = data.get("meal_time_hr", 2)
        features["motion_artifact"] = data.get("motion_artifact", 0)

        X = pd.DataFrame([features])

        for col in features_order:
            if col not in X.columns:
                X[col] = 0

        X = X[features_order]
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)

        pred = model.predict(X)[0]

        print("PREDICTED:", pred)

        return jsonify({
            "predicted_glucose": round(float(pred), 1)
        }), 200

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
    )
