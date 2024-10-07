# import modules
import os

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from sklearn.model_selection import train_test_split

from read_data import read_garaje, read_hexa, read_quadcarbono, read_uav_fd


class affine_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        source_path,
        target_path,
        source_fs,
        target_fs,
        section_len=10,
        window_len=1,
        overlap=0.5,
        batch_size=32,
        num_workers=12,
    ):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.source_fs = source_fs
        self.target_fs = target_fs
        self.section_len = section_len
        self.window_len = window_len
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = [0.8, 0.2]

    def prepare_data(self):
        assert os.path.exists(
            self.source_path
        ), f"Path {self.source_path} does not exist"
        assert os.path.exists(
            self.target_path
        ), f"Path {self.target_path} does not exist"

        if "quadcarbono" in self.source_path:
            source_dataset = read_quadcarbono(
                self.source_path, self.source_fs, self.section_len
            )
        elif "UAV_FD" in self.source_path:
            source_dataset = read_uav_fd(
                self.source_path, self.source_fs, self.section_len
            )
        elif "garage_manual" in self.source_path:
            source_dataset = read_garaje(
                self.source_path, self.source_fs, self.section_len
            )
        elif "hexaF550" in self.source_path:
            source_dataset = read_hexa(
                self.source_path, self.source_fs, self.section_len
            )
        else:
            raise ValueError("Source path not recognized")

        if "quadcarbono" in self.target_path:
            target_dataset = read_quadcarbono(
                self.target_path, self.target_fs, self.section_len
            )
        elif "UAV_FD" in self.target_path:
            target_dataset = read_uav_fd(
                self.target_path, self.target_fs, self.section_len
            )
        elif "garage_manual" in self.target_path:
            target_dataset = read_garaje(
                self.target_path, self.target_fs, self.section_len
            )
        elif "hexaF550" in self.target_path:
            target_dataset = read_hexa(
                self.target_path, self.target_fs, self.section_len
            )

        else:
            raise ValueError("Target path not recognized")

        # Train test split
        self.train, self.val = train_test_split(
            source_dataset,
            test_size=self.split_ratio[1] / (self.split_ratio[0] + self.split_ratio[1]),
            random_state=42,
        )
        self.test = target_dataset

        # split into windows of length window_len with overlap
        print("Splitting into windows")
        self.train = self.split_windows(self.train, self.source_fs)
        self.val = self.split_windows(self.val, self.source_fs)
        self.test = self.split_windows(self.test, self.target_fs)

        print("Calculating welch energy bands")
        self.energy_train, self.label_train = self.calculate_energy_bands(self.train)
        self.energy_val, self.label_val = self.calculate_energy_bands(self.val)
        self.energy_test, self.label_test = self.calculate_energy_bands(self.test)
        self.energy_test_orig = self.energy_test.copy()

        self.scale_factor = self.fit_transport_matrix(
            self.energy_train,
            self.energy_test,
        )

        # self.scale_factor = 1.0  # no scaling
        self.scale_factor = 1.12  # quadcarbono to quadcarbono1000
        # self.scale_factor = 0.734 # garaje to quadcarbono

        print(f"scale_factor: {self.scale_factor}")

        # transport the test distribution
        self.energy_test, self.label_test = self.calculate_energy_bands(
            self.test, self.scale_factor
        )

        # normalize by category
        source_norm_factor = 2e1 * np.max(
            np.mean(self.energy_train[self.label_train == 0], axis=0), axis=0
        )
        self.energy_train = 3 * self.energy_train / source_norm_factor
        self.energy_val = 3 * self.energy_val / source_norm_factor

        target_norm_factor = 2e1 * np.max(
            np.mean(self.energy_test[self.label_test == 0], axis=0), axis=0
        )
        # print(f"target_norm_factor: {target_norm_factor}")
        self.energy_test = 3 * self.energy_test / target_norm_factor

        target_norm_factor_orig = 2e1 * np.max(
            np.mean(self.energy_test_orig, axis=0), axis=0
        )
        self.energy_test_orig = 3 * self.energy_test_orig / target_norm_factor_orig

        # labels 0, 5, 10 to 0,1,2
        self.label_train = self.label_train // 5
        self.label_val = self.label_val // 5
        self.label_test = self.label_test // 5

    def calculate_energy_bands(self, data, scale_factor=1):
        # calculate welch energy for each window
        label = data[:, 0, -1]
        f, energy_welch = signal.welch(
            data[:, :, :-1], 350, nperseg=70 // scale_factor, axis=1
        )
        if scale_factor <= 1:
            energy = energy_welch[:, :36, :]
        else:
            last_row = energy_welch[:, -1, :]
            # repeat last row to fill the array up to size 36
            last_row = np.repeat(
                last_row[:, np.newaxis, :], 36 - energy_welch.shape[1], axis=1
            )
            energy = np.concatenate((energy_welch, last_row), axis=1)

        return energy.astype(np.float32), label.astype(np.int8)

    def split_windows(self, data, fs):
        # split into windows of length window_len with overlap using torch Unfold
        # data is a list of numpy arrays the shape of each array is (section_len * fs, 37)
        # the resulting tensor will have shape (windows, window_len * fs, 37)
        # the last column is the category

        res = np.empty((0, self.window_len * fs, data[0].shape[1]))
        for d in data:
            d = torch.from_numpy(d)
            d = d.unfold(
                0,
                self.window_len * fs,
                int(self.window_len * fs * (1 - self.overlap)),
            )
            d = d.swapaxes(1, 2)
            d = d.numpy()
            res = np.vstack((res, d))

        return res

    def fit_transport_matrix(self, source, target):
        # fit affine scalings between source and target distributions
        self.T_unnormed = np.ones((source.shape[1], target.shape[1], source.shape[2]))

        # average data over windows (axis 0)
        source = np.mean(source, axis=0)
        target = np.mean(target, axis=0)

        source_peak = np.argmax(source[:, 0])
        target_peak = np.argmax(target[:, 0])
        scale_factor = target_peak / source_peak

        return scale_factor

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            list(zip(self.energy_train, self.label_train)),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            list(zip(self.energy_val, self.label_val)),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            list(zip(self.energy_test, self.label_test)),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def plot_distributions(self):
        # to ensure that the domain adaptation worked, plot the distributions before and after
        # the transport and source and target
        # print(f"energy test shape: {self.energy_test.shape}")
        x = np.arange(
            self.energy_test.shape[1] * 3.17142857143,
            step=3.17142857143,
            dtype=np.float64,
        )

        # manually plot arrows from the source to the target distribution
        plt.figure(1, figsize=(6, 2), dpi=300)

        plt.plot(
            x,
            np.mean(self.energy_train[self.label_train == 0], axis=0)[:, 0]
            / np.max(np.mean(self.energy_train[self.label_train == 0], axis=0)[:, 0]),
            "xkcd:kelly green",
            label="Domain 1",
        )

        plt.plot(
            x,
            np.mean(self.energy_test[self.label_test == 0], axis=0)[:, 0]
            / np.max(np.mean(self.energy_test[self.label_test == 0], axis=0)[:, 0]),
            "xkcd:orange",
            ls="--",
            # label="Domain 2 -> 1",
            label=r"Domain 2 $\rightarrow$ 1",
        )

        plt.plot(
            x,
            np.mean(self.energy_test_orig[self.label_test == 0], axis=0)[:, 0]
            / np.max(
                np.mean(self.energy_test_orig[self.label_test == 0], axis=0)[:, 0]
            ),
            "xkcd:purple",
            label="Domain 2",
        )

        plt.gca().spines[["top", "right"]].set_visible(False)
        plt.legend()
        plt.gca().spines["bottom"].set_position(("data", 0))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Energy")
        plt.yticks([])

        plt.show()

    def plot_sensibility(self):
        # to ensure that the domain adaptation worked, plot the distributions before and after
        # the transport and source and target
        # print(f"energy test shape: {self.energy_test.shape}")
        x = np.arange(
            self.energy_test.shape[1] * 3.17142857143,
            step=3.17142857143,
            dtype=np.float64,
        )

        # manually plot arrows from the source to the target distribution
        plt.figure(1, figsize=(6, 6), dpi=300)

        plt.plot(
            x,
            np.mean(self.energy_train[self.label_train == 0], axis=0)[:, 0]
            / np.max(np.mean(self.energy_train[self.label_train == 0], axis=0)[:, 0]),
            "xkcd:kelly green",
            label=f"{self.source_path} nominal",
        )

        plt.plot(
            x,
            np.mean(self.energy_train[self.label_train == 2], axis=0)[:, 0]
            / np.max(np.mean(self.energy_train[self.label_train == 0], axis=0)[:, 0]),
            "xkcd:kelly green",
            label=f"{self.source_path} damaged",
            ls="--",
        )

        plt.plot(
            x,
            np.mean(self.energy_test_orig[self.label_test == 0], axis=0)[:, 0]
            / np.max(
                np.mean(self.energy_test_orig[self.label_test == 0], axis=0)[:, 0]
            ),
            "xkcd:purple",
            label=f"{self.target_path} nominal",
        )

        plt.plot(
            x,
            np.mean(self.energy_test_orig[self.label_test == 2], axis=0)[:, 0]
            / np.max(
                np.mean(self.energy_test_orig[self.label_test == 0], axis=0)[:, 0]
            ),
            "xkcd:purple",
            label=f"{self.target_path} damaged",
            ls="--",
        )
        # horizontal line at 1
        plt.axhline(1, color="grey", lw=0.4, ls="--")

        plt.gca().spines[["top", "right"]].set_visible(False)
        plt.legend()
        plt.gca().spines["bottom"].set_position(("data", 0))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Energy")
        plt.yticks([])

        plt.show()
