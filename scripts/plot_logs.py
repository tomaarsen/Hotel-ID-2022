import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import glob

pattern = re.compile(r"Epoch (?P<epoch>\d+):.*\[(?P<time>[^,]+),\s+(?P<speed>[^,]+)(it/s|s/it),\s+loss=(?P<loss>[^,]+),\s+v_num=(?P<v_num>[^,]+),\s+val_acc=(?P<val_acc>[^,]+),\s+val_loss=(?P<val_loss>[^,]+),\s+train_acc=(?P<train_acc>[\d\.]+)]")
goal_path = "logs/pngs"
if not os.path.isdir(goal_path):
    os.mkdir(goal_path)

for filename in glob.glob("./logs/slurm/*.out"):
    version, _ = os.path.splitext(os.path.basename(filename))
    if version == "%J":
        continue
    goal_file = os.path.join(goal_path, f"slurm_logs_{version}.png")
    if os.path.isfile(goal_file):
        continue

    print(f"Parsing SLURM logs for Version ({version})...")
    with open(filename, "r", encoding="utf8") as f:
        data = f.read()

    logs = defaultdict(list)
    for match in pattern.finditer(data):
        for key, value in match.groupdict().items():
            try:
                logs[key].append(float(value))
            except:
                logs[key].append(value)
    print(f"Parsed SLURM logs for Version ({version})!")

    if "val_acc" not in logs or len(logs["val_acc"]) < 50:
        continue

    # Create 3 subplots, one for loss, one for accuracy
    fig, (loss_ax, acc_ax) = plt.subplots(2, 1, figsize=(4.8*1.5, 4.8*2))
    fig.suptitle(f"Training vs Validation for Version ({version})")

    # Loss subplot
    loss_ax.plot("epoch", "loss", data=logs, label="Training Loss")
    loss_ax.plot("epoch", "val_loss", data=logs, label="Validation Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()

    # Accuracy subplot
    acc_ax.plot("epoch", "train_acc", data=logs, label="Training Accuracy")
    acc_ax.plot("epoch", "val_acc", data=logs, label="Validation Accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy (0 to 1)")
    acc_ax.legend()

    fig.tight_layout()
    fig.savefig(goal_file)

print(f"Training vs Validation loss png files generated (see {goal_path})")
print("Consider downloading the generated png files with:")
print(f"> scp <user>@cn99.science.ru.nl:{goal_path}/* logs/pngs/")
print("From your local machine, and then viewing them there.")
