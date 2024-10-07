# %%
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import confusion_matrix

from affine import affine_DataModule
from fd_datamodule import FD_DataModule


# %% model definition
class Model(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=256, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long())
        self.log("train_loss", loss)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y) / len(y)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long())
        self.log("val_loss", loss)
        # log accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y) / len(y)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y.long())
        self.log("test_loss", loss)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y) / len(y)
        self.log("test_acc", acc)

    def predict(self, x):
        return self(x)

    # log graph of model and histogram of weights
    def on_train_epoch_end(self):
        # log graph
        self.logger.experiment.add_graph(self, torch.zeros(1, self.input_size))

        # log histogram of weights
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)


# %% train

garaje_datamodule = FD_DataModule(
    data_path="data/garage_manual/",
    fs=222,
)

affine_datamodule = affine_DataModule(
    source_path="data/quadcarbono/",
    target_path="data/quadcarbono_1000/",
    source_fs=222,
    target_fs=222,
)

# Decide which module to use for training and testing and experiment name
experiment_name = "Experiment1"

data_module = affine_datamodule  # CHANGE_HERE for the desired dataset
data_module.prepare_data()

model = Model(input_size=36 * 6, output_size=3)
trainer = pl.Trainer(
    max_epochs=300,
    accelerator="auto",
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
    default_root_dir=f"logs/{experiment_name}",
)

trainer.fit(model, data_module)


# %% test, confusion matrix

y_true = []
y_pred = []
for x, y in data_module.test_dataloader():
    y_true.extend(y)
    y_pred.extend(torch.argmax(model(x), dim=1))


cm = confusion_matrix(y_true, y_pred)
# nice looking confusion matrix
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
plt.figure(figsize=(3.5, 3.5), dpi=200)

sns.heatmap(
    cm,
    annot=np.array([
        f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)
    ]).reshape(2, 2),
    fmt="",
    cmap="Blues",
    cbar=False,
)
plt.xlabel("Predicted")
plt.ylabel("True")
class_names = ["Normal", "Damaged"]
plt.xticks([0.5, 1.5], class_names)
plt.yticks([0.5, 1.5], class_names)

# add perimeter lines on the confusion matrix
plt.plot([0, 0], [0, 2], "k", linewidth=2)
plt.plot([0, 2], [2, 2], "k", linewidth=2)
plt.plot([2, 2], [2, 0], "k", linewidth=2)
plt.plot([2, 0], [0, 0], "k", linewidth=2)

# add accuracy
accuracy = np.trace(cm) / np.sum(cm)
precision = cm[1, 1] / np.sum(cm[:, 1])
recall = cm[1, 1] / np.sum(cm[1, :])
plt.text(0, -0.05, f"Accuracy: {accuracy:.4f}", fontsize=10)
plt.text(0, -0.15, f"Precision: {precision:.4f}", fontsize=10)
plt.text(0, -0.25, f"Recall:      {recall:.4f}", fontsize=10)


# %% additional plots
if hasattr(data_module, "plot_sensibility"):
    data_module.plot_sensibility()

if hasattr(data_module, "plot_split"):
    data_module.plot_split()
