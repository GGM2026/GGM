import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from natsort import natsorted

from data_provider.uea import normalize_batch_ts


class PTBXLLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature")
        self.label_path = os.path.join(root_path, "Label", "label.npy")

        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_data(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)

        no_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        mi_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        sttc_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])
        cd_list = list(data_list[np.where(data_list[:, 0] == 3)][:, 1])
        hyp_list = list(data_list[np.where(data_list[:, 0] == 4)][:, 1])

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_data(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []

        subject_label = np.load(label_path)
        filenames = natsorted(
            [f for f in os.listdir(data_path) if f.endswith(".npy")]
        )

        flag = flag.upper() if flag is not None else None
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("PTB-XL train ids:", sorted(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("PTB-XL val ids:", sorted(ids))
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("PTB-XL test ids:", sorted(ids))
        else:
            ids = set(subject_label[:, 1].tolist())
            print("PTB-XL all ids:", sorted(ids))

        for j, filename in enumerate(filenames):
            trial_label = subject_label[j]
            pid = int(trial_label[1])

            if pid not in ids:
                continue

            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)

            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        X = np.array(feature_list, dtype=np.float32)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0].astype(np.int64)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.X[index]),
            torch.tensor(self.y[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.y)


class ADFTDLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature")
        self.label_path = os.path.join(root_path, "Label", "label.npy")

        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self.load_train_val_test_list(
            self.label_path, a, b
        )

        self.X, self.y = self.load_data(self.data_path, self.label_path, flag=flag)

        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def load_train_val_test_list(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)

        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        ftd_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        ad_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])

        train_ids = (
            hc_list[: int(a * len(hc_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            hc_list[int(b * len(hc_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids

    def load_data(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []

        subject_label = np.load(label_path)
        filenames = natsorted(
            [f for f in os.listdir(data_path) if f.endswith(".npy")]
        )

        flag = flag.upper() if flag is not None else None
        if flag == "TRAIN":
            ids = set(self.train_ids)
            print("ADFTD train ids:", sorted(ids))
        elif flag == "VAL":
            ids = set(self.val_ids)
            print("ADFTD val ids:", sorted(ids))
        elif flag == "TEST":
            ids = set(self.test_ids)
            print("ADFTD test ids:", sorted(ids))
        else:
            ids = set(subject_label[:, 1].tolist())
            print("ADFTD all ids:", sorted(ids))

        for j, filename in enumerate(filenames):
            trial_label = subject_label[j]
            pid = int(trial_label[1])

            if pid not in ids:
                continue

            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)

            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)

        X = np.array(feature_list, dtype=np.float32)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)

        return X, y[:, 0].astype(np.int64)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.X[index]),
            torch.tensor(self.y[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.y)