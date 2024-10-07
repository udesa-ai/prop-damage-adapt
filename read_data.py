import os

import numpy as np


def read_garaje(path, fs, section_len):
    dataset = []
    for file in os.listdir(path):
        if file.endswith("_inertial.csv"):
            print(f"Reading {file} and splitting into sections")
            # Read data direcly into a np array
            data = np.genfromtxt(path + file, delimiter=",")

            # Drop header
            data = data[1:, :]

            # drop column 1
            data = data[:, 1:]

            # add category column
            name = file.split("_")[1]
            name_2_cat = {"H0": 0, "H11": 5, "H12": 10}

            data = np.hstack((data, np.full((data.shape[0], 1), name_2_cat[name])))

            # split into sections
            data = np.array_split(
                data, data.shape[0] // (section_len * fs)
            )
            dataset.extend(data)
    return dataset

def read_uav_fd(path, fs, section_len):
    dataset = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            print(f"Reading {file} and splitting into sections")
            # Read data direcly into a np array
            data = np.genfromtxt(path + file, delimiter=",")
            # Drop header
            data = data[1:, :]
            # search for nan values
            assert not np.isnan(data).any(), f"NaN values in {file}"

            # keep only the columns we need
            keep_columns = [0, 1, 2, 3, 4, 5, 36]
            data = data[:, keep_columns]

            # split into sections
            data = np.array_split(
                data, data.shape[0] // (section_len * fs)
            )
            dataset.extend(data)
    return dataset

def read_quadcarbono(path, fs, section_len):
    dataset = []
    for file in os.listdir(path):
        if file.endswith("_inertial.csv"):
            print(f"Reading {file} and splitting into sections")
            # Read data direcly into a np array
            data = np.genfromtxt(path + file, delimiter=",")

            # Drop header
            data = data[1:, :]

            # drop column 1
            data = data[:, 1:]

            # add category column
            name = file.split("_")[1]
            name_2_cat = {"Hnominal": 0, "Hasymm82": 5, "Hasymm164": 10}

            data = np.hstack((data, np.full((data.shape[0], 1), name_2_cat[name])))

            # split into sections
            data = np.array_split(
                data, data.shape[0] // (section_len * fs)
            )
            dataset.extend(data)
    return dataset

def read_hexa(path, fs, section_len):
    dataset = []
    for file in os.listdir(path):
        if file.endswith("_inertial.csv"):
            print(f"Reading {file} and splitting into sections")
            # Read data direcly into a np array
            data = np.genfromtxt(path + file, delimiter=",")

            # Drop header
            data = data[1:, :]

            # drop column 1
            data = data[:, 1:]

            # add category column
            name = file.split("_")[1]
            name_2_cat =  {"Hnominal": 0, "Hasymm10": 10}

            data = np.hstack((data, np.full((data.shape[0], 1), name_2_cat[name])))

            # split into sections
            data = np.array_split(
                data, data.shape[0] // (section_len * fs)
            )
            dataset.extend(data)
    return dataset