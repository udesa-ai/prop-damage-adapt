"""Reads M1_NAME_intertial.csv and augments the data by mirroring in all four dimensions
mot="1"; # M1 (original)
ax_new =  ax;
ay_new =  ay;
gx_new =  gx;
gy_new =  gy;

mot="2"; # M2
ax_new = -ay;
ax_new =  ax;
gx_new = -gy;
gy_new =  gx;

mot="3"; # M3
ax_new = -ax;
ay_new = -ay;
gx_new = -gx;
gy_new = -gy;


mot="4"; # M4
ax_new =  ay;
ay_new = -ax;
gx_new =  gy;
gy_new = -gx;

file header:
timestamp,accelerometer_m_s2_1,accelerometer_m_s2_2,accelerometer_m_s2_3,gyro_rad_1,gyro_rad_2,gyro_rad_3
"""

import os

import numpy as np
import pandas as pd

NAME = "Hnominal_corto"

# Read the data
data = pd.read_csv(f"M1_{NAME}_inertial.csv")
data.columns = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]

# Create the new data
data2 = data.copy()
data3 = data.copy()
data4 = data.copy()

# Mirror the data
data2["ax"] = -data["ay"]
data2["ay"] = data["ax"]
data2["gx"] = -data["gy"]
data2["gy"] = data["gx"]

data3["ax"] = -data["ax"]
data3["ay"] = -data["ay"]
data3["gx"] = -data["gx"]
data3["gy"] = -data["gy"]

data4["ax"] = data["ay"]
data4["ay"] = -data["ax"]
data4["gx"] = data["gy"]
data4["gy"] = -data["gx"]

# Use original header
# timestamp,accelerometer_m_s2_1,accelerometer_m_s2_2,accelerometer_m_s2_3,gyro_rad_1,gyro_rad_2,gyro_rad_3
data2.columns = ["timestamp", "accelerometer_m_s2_1", "accelerometer_m_s2_2", "accelerometer_m_s2_3", "gyro_rad_1", "gyro_rad_2", "gyro_rad_3"]
data3.columns = ["timestamp", "accelerometer_m_s2_1", "accelerometer_m_s2_2", "accelerometer_m_s2_3", "gyro_rad_1", "gyro_rad_2", "gyro_rad_3"]
data4.columns = ["timestamp", "accelerometer_m_s2_1", "accelerometer_m_s2_2", "accelerometer_m_s2_3", "gyro_rad_1", "gyro_rad_2", "gyro_rad_3"]


# Save the data
data2.to_csv(f"M2_{NAME}_inertial.csv", index=False)
data3.to_csv(f"M3_{NAME}_inertial.csv", index=False)
data4.to_csv(f"M4_{NAME}_inertial.csv", index=False)


