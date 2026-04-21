import sys
from pathlib import Path
import re
import pandas as pd




# Ensure the library path is in sys.path
lib_path = Path.cwd().parent.resolve()

if str(lib_path) not in sys.path:
    sys.path.insert(0, str(lib_path))

import lib

BLINKING = False
#path = lib.sdownload(id=6, local_folder="data", blinking=BLINKING)

current_path = Path.cwd()
path = current_path / "csv_generats"


csv_files = sorted(path.glob("*CM*.csv"), key=lambda x: int(re.findall(r'\d+', x.stem)[0]))
training_indices = [1,2,3]
training_paths = [csv_files[i - 1] for i in training_indices]
validation_paths = [path for path in csv_files if path not in training_paths]

import pandas as pd


training_clusters, training_backgrounds = [], []
for csv_path in training_paths:
    df = pd.read_csv(csv_path)

    # Normalize coordinates to [0, 1]
    coords = df[["x", "y"]]
    mins, maxs = coords.min(), coords.max()
    df[["x", "y"]] = (coords - mins) / (maxs - mins)

    # Group by 'index'
    for idx, group in df.groupby("index"):
        # 0 (the background)
        if idx == 0:
            training_backgrounds.append(group[["x", "y"]].values)
        else:
            centered = group[["x", "y"]] - group[["x", "y"]].mean()
            training_clusters.append(centered)

metadata = {
    "csv_generats": {
        # "training_indices": list(range(1, 4)),
        "builder": "GraphDataset",
        "builder_kwargs": {
            "dataset_size": 1500,
            "noise_point_range": [1200, 1500],
            "cluster_count_range": [5, 10],
            "connectivity_radius": 0.1,
            "cluster_sampling_range": [0.8, 1.0],
        },

        "recurrent_iterations": 20,
        "checkpoint": "checkpoint.pt",
    }
}

metadata = metadata[path.name]

# Initialize the MIRO builder with the training clusters and metadata
builder_args = (training_clusters,)
if BLINKING:
    builder_args += (training_backgrounds,)
    
builder = getattr(lib, metadata["builder"])(
    *builder_args, **metadata["builder_kwargs"]
)
augmented_dataset = builder()


import deeplay as dl
from torch_geometric.loader import DataLoader

clusterer = dl.MIRO(
    num_outputs=2,  # Number of output features (e.g., x, y displacements)
    connectivity_radius=builder.connectivity_radius,  # Radius for graph connectivity (matches dataset)
    num_iterations=metadata["recurrent_iterations"],  # Number of iterations for graph processing
)
clusterer = clusterer.create()


train_loader = DataLoader(
    dataset=augmented_dataset,  # The dataset to be loaded
    batch_size=1,  # Number of samples per batch
    shuffle=True,  # Shuffle the dataset at every epoch
)

# Initialize the trainer
trainer = dl.Trainer(max_epochs=30)  # Maximum number of training epochs

trainer.fit(
    clusterer,  # The MIRO model to be trained
    train_loader,  # The DataLoader providing the training data
)