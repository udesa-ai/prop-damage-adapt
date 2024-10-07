import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import signal
from sklearn.model_selection import train_test_split

from read_data import read_garaje, read_quadcarbono, read_uav_fd


class FD_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        fs,
        section_len=10,
        window_len=1,
        overlap=0.5,  # % overlap
        batch_size=32,
        num_workers=12,
        split=[0.78, 0.02, 0.20],
    ):
        super().__init__()
        self.path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fs = fs
        self.section_len = section_len
        self.split_ratio = split
        self.window_len = window_len
        self.overlap = overlap

    def prepare_data(self):
        if "quadcarbono" in self.path:
            dataset = read_quadcarbono(self.path, self.fs, self.section_len)
        elif "UAV_FD" in self.path:
            dataset = read_uav_fd(self.path, self.fs, self.section_len)
        elif "garage_manual" in self.path:
            dataset = read_garaje(self.path, self.fs, self.section_len)
        else:
            raise ValueError("Path not recognized")

        # Train test split
        dev, self.test = train_test_split(
            dataset, test_size=self.split_ratio[2], random_state=42
        )
        self.train, self.val = train_test_split(
            dev,
            test_size=self.split_ratio[1] / (self.split_ratio[0] + self.split_ratio[1]),
            random_state=42,
        )

        # split into windows of length window_len with overlap
        print("Splitting into windows")
        self.train = self.split_windows(self.train)
        self.val = self.split_windows(self.val)
        self.test = self.split_windows(self.test)

        print("Calculating welch energy bands")
        self.energy_train, self.label_train = self.calculate_energy_bands(self.train)
        self.energy_val, self.label_val = self.calculate_energy_bands(self.val)
        self.energy_test, self.label_test = self.calculate_energy_bands(self.test)


        # # remove first 23 bands TODO: remove this
        # self.energy_train[:, :23, :] = 0
        # self.energy_val[:, :23, :] = 0
        # self.energy_test[:, :23, :] = 0


        # normalize by category
        norm_factor = 2e1 * np.max(np.mean(self.energy_train, axis=0), axis=0)

        self.energy_train = 3 * self.energy_train / norm_factor
        self.energy_val = 3 * self.energy_val / norm_factor
        self.energy_test = 3 * self.energy_test / norm_factor

        # labels 0, 5, 10 to 0,1,2
        self.label_train = self.label_train // 5
        self.label_val = self.label_val // 5
        self.label_test = self.label_test // 5

    def calculate_energy_bands(self, data):
        # calculate welch energy for each window
        label = data[:, 0, -1]
        f, energy = signal.welch(data[:, :, :-1], 350, nperseg=70, axis=1)

        return energy.astype(np.float32), label.astype(np.int8)

    def split_windows(self, data):
        # split into windows of length window_len with overlap using torch Unfold
        # data is a list of numpy arrays the shape of each array is (section_len * fs, 37)
        # the resulting tensor will have shape (windows, window_len * fs, 37)
        # the last column is the category

        res = np.empty((0, self.window_len * self.fs, data[0].shape[1]))
        for d in data:
            d = torch.from_numpy(d)
            d = d.unfold(
                0,
                self.window_len * self.fs,
                int(self.window_len * self.fs * (1 - self.overlap)),
            )
            d = d.swapaxes(1, 2)
            d = d.numpy()
            res = np.vstack((res, d))

        return res

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

    def plot_split(self):
        # plot the split of the data with respect to the category
        # last column is the category
        train = self.train[:, 0, :]
        val = self.val[:, 0, :]
        test = self.test[:, 0, :]

        train = pd.DataFrame(train)
        val = pd.DataFrame(val)
        test = pd.DataFrame(test)

        # add label column
        train["label"] = "train"
        val["label"] = "val"
        test["label"] = "test"

        # concatenate
        df = pd.concat([train, val, test])

        # drop all columns except the label and the category
        df = df[[df.columns[-2], df.columns[-1]]]
        df.columns = ["category", "label"]

        # plot
        sns.set_theme(style="darkgrid")
        sns.countplot(x="label", hue="category", data=df)
        # plt.show()

    def plot_distributions(self):
        # plot the distribution of the data with respect to the category
        x = np.arange(self.energy_test.shape[1], dtype=np.float64)
        n = self.energy_test.shape[-1]

        # subplots
        fig, axs = plt.subplots(n, 1, sharex=True, figsize=(10, 10))
        fig.suptitle("Energy bands distribution")

        for i in range(n):
            axs[i].stem(
                x,
                np.mean(self.energy_test[self.label_test == 0], axis=0)[:, i],
                "b",
                label="Source distribution no fail",
            )
            axs[i].plot(
                x,
                np.mean(self.energy_test[self.label_test == 1], axis=0)[:, i],
                "royalblue",
                label=r"Source distribution 5% fail",
            )
            axs[i].plot(
                x,
                np.mean(self.energy_test[self.label_test == 2], axis=0)[:, i],
                "navy",
                label=r"Source distribution 10% fail",
                ls="--",
            )
            axs[i].set_ylabel(f"Band {i+1}")
            axs[i].legend()
        axs[-1].set_xlabel("Frequency [Hz]")
        plt.show()
