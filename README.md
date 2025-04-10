Code for "Propeller Damage Detection: Adapting Models to Diverse UAV Sizes" Submitted to IEEE Robotics and Automation Letters.

## Abstract
This manuscript introduces a transfer learning method for adapting propeller fault detection neural networks to different unmanned aerial vehicles (UAVs). After training a simple model for detecting if any propeller in a specific vehicle has a failure (in this case, a chipped tip), a domain adaptation based in the vehicles' physics is performed in order to use the same model to detect failures in vehicles with different structures, weights, or motor-propeller sets. A key feature is that the detection model uses only inertial sensors that are standard in commercial UAVs, making it broadly applicable without the need for additional hardware.

## Video
[![Video thumbnail](https://github.com/user-attachments/assets/1f02103a-21a9-4bab-a56d-0cc59ae2c990)](https://www.youtube.com/watch?v=MXNc1VfQ0Y8)
https://www.youtube.com/watch?v=MXNc1VfQ0Y8

## To run

1. Clone this repository.

1. Install Python3 dependencies, `matplotlib`, `numpy`, `pytorch`, `scikit-learn`, `scipy`, `seaborn`, `torch`, `lightning`, `tqdm`

1. Run `model_train.py` with the following configuration for each experiment:

### Experiments

#### Experiment VI.A

Use the `quadcarbono_datamodule` inside `model_train.py`
```python
data_module = quadcarbono_datamodule
```

#### Experiment VI.B
Use the `affine_datamodule` inside `model_train.py`
```python
affine_datamodule = affine_DataModule(
    source_path="data/garage_manual/",
    target_path="data/quadcarbono/",
    source_fs=222,
    target_fs=222,
)
data_module = affine_datamodule
```

#### Experiment VI.B.1
For the naive use no scaling, set `self.scale_factor = 1.0` inside `affine.py`

#### Experiment VI.B.2
For the scaled experiment, set `self.scale_factor = 0.734` inside `affine.py`

#### Experiment VI.B.3
For this experiment use the data in `quadcarbono_weight_shift` as test instead of `Hasymm164_inertial`

#### Experiment VI.C 
Use the `affine_datamodule` inside `model_train.py`
```python
affine_datamodule = affine_DataModule(
    source_path="data/quadcarbono/",
    target_path="data/quadcarbono_1000/",
    source_fs=222,
    target_fs=222,
)
data_module = affine_datamodule
```

#### Experiment VI.C.1
For the naive use no scaling, set `self.scale_factor = 1.0` inside `affine.py`

#### Experiment VI.C.2
For the scaled experiment, set `self.scale_factor = 1.12` inside `affine.py`

#### Experiment VI.C.3
For this experiment use `quadcarbono_water` as test 

#### Experiment VI.D
```python
affine_datamodule = affine_DataModule(
    source_path="data/quadcarbono/",
    target_path="data/hexaF550/",
    source_fs=222,
    target_fs=222,
)
data_module = affine_datamodule
```



