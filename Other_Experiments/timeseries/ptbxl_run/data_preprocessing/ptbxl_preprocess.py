import os
import numpy as np
import pandas as pd
import wfdb
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

ROOT = "/nfs/hpc/share/pourmans/timeseries-ggd/Medformer/ptb-xl"
RECORDS_PATH = os.path.join(ROOT, "records500")

OUT_ROOT = "/nfs/hpc/share/pourmans/timeseries-ggd/Medformer/PTB-XL"
FEATURE_DIR = os.path.join(OUT_ROOT, "Feature")
LABEL_DIR = os.path.join(OUT_ROOT, "Label")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)


def final_scp(codes: str) -> str:
    d = {}
    ls = codes.strip("{").strip("}").split(",")
    for code in ls:
        k = code.split(":")[0].replace("'", "").replace(" ", "")
        v = float(code.split(":")[-1])
        d[k] = v

    scp = max(d, key=d.get)

    if scp in {
        "NDT", "NST_", "DIG", "ISC_", "ISCAL", "LNGQT", "ISCIN",
        "ISCIL", "ISCAS", "ISCLA", "ANEUR", "EL", "ISCAN"
    }:
        return "STTC"
    elif scp == "NORM":
        return "NORM"
    elif scp in {
        "IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI",
        "INJAL", "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL"
    }:
        return "MI"
    elif scp in {"LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP"}:
        return "HYP"
    elif scp in {"LAFB", "IRBBB", "1AVB", "IVCD", "CRBBB", "CLBBB",
                 "LPFB", "WPW", "ILBBB", "3AVB", "2AVB"}:
        return "CD"
    else:
        return "others"


def resample_to_250hz(x: np.ndarray, orig_freq=500) -> np.ndarray:
    t = np.linspace(1, len(x), len(x))
    f = interpolate.interp1d(t, x, kind="linear")
    t_new = np.linspace(1, len(x), int(len(x) / orig_freq * 250))
    return f(t_new)


def normalize_trial(trial_2d: np.ndarray) -> np.ndarray:
    # trial_2d shape: (time, channels)
    scaler = StandardScaler()
    return scaler.fit_transform(trial_2d)


# Load metadata
info = pd.read_csv(os.path.join(ROOT, "ptbxl_database.csv"), index_col=None)
info = info[["ecg_id", "scp_codes", "patient_id"]]
info["scp_codes"] = info["scp_codes"].apply(final_scp)

# Keep only patients with one consistent diagnosis across trials, excluding "others"
id_dict = {}
order = 1
grouped = info.groupby("patient_id", sort=True)

for _, df in grouped:
    scps = df["scp_codes"].tolist()
    if ("others" not in set(scps)) and (len(set(scps)) == 1):
        pid = f"{order:05d}"
        id_dict[pid] = [df["ecg_id"].tolist(), scps]
        order += 1

print("Kept patients:", len(id_dict))

# Build features
for pid, (ecg_ids, scps) in id_dict.items():
    subject_trials = []

    for folder in os.listdir(RECORDS_PATH):
        folder_path = os.path.join(RECORDS_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.endswith(".hea"):
                continue

            ecg_id = int(fname.split(".")[0].split("_")[0])
            if ecg_id not in ecg_ids:
                continue

            rec_path = os.path.join(folder_path, fname[:-4])  # strip .hea
            ecg_data, fields = wfdb.rdsamp(rec_path)  # shape (5000, 12) at 500 Hz

            # resample each channel to 250 Hz -> shape (2500, 12)
            channels = []
            for ch in range(ecg_data.shape[1]):
                ch_data = resample_to_250hz(ecg_data[:, ch], orig_freq=500)
                channels.append(ch_data)
            trial = np.array(channels).T

            # standardize per trial
            trial = normalize_trial(trial)

            subject_trials.append(trial)

    subject_trials = np.array(subject_trials)  # (num_trials, 2500, 12)

    # split each 10-second trial into ten 1-second windows: (num_trials*10, 250, 12)
    subject_windows = subject_trials.reshape(-1, 250, subject_trials.shape[-1])

    np.save(os.path.join(FEATURE_DIR, f"feature_{pid}.npy"), subject_windows)

# Build labels
labels = []
for pid, (_, scps) in id_dict.items():
    scp = list(set(scps))[0]
    if scp == "NORM":
        diag = 0
    elif scp == "MI":
        diag = 1
    elif scp == "STTC":
        diag = 2
    elif scp == "CD":
        diag = 3
    else:
        diag = 4  # HYP
    labels.append([diag, int(pid)])

labels = np.array(labels, dtype=np.int64)
np.save(os.path.join(LABEL_DIR, "label.npy"), labels)

print("Saved:")
print(" -", FEATURE_DIR)
print(" -", os.path.join(LABEL_DIR, "label.npy"))
print("Example label shape:", labels.shape)