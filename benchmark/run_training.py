import sys
from pathlib import Path
import re
import pandas as pd
from pathlib import Path
import pickle
from torch_geometric.loader import DataLoader



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
training_indices = [1,2,3,4,5,6,7,8,9,10,11,12]
training_paths = [csv_files[i - 1] for i in training_indices]
validation_paths = [path for path in csv_files if path not in training_paths]




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
        "training_indices": list(range(1, 6)),
        "builder": "GraphDataset",
        "builder_kwargs": {
            "dataset_size": 1500,
            "noise_point_range": [450, 550],
            "cluster_count_range": [10, 20],
            "connectivity_radius": 0.2,
        },

        "recurrent_iterations": 20,
        "checkpoint": "checkpoint.pt",
    }
}

metadata = metadata[path.name]

# ---- cache control ----
DATA_CACHE_PATH = Path("augmented_dataset.pkl")
LOAD_IF_AVAILABLE = True   # if True, load from disk when possible
FORCE_REGENERATE = False   # if True, ignore cache and rebuild

# Initialize the MIRO builder with the training clusters and metadata
builder_args = (training_clusters,)
if BLINKING:
    builder_args += (training_backgrounds,)

builder = getattr(lib, metadata["builder"])(
    *builder_args, **metadata["builder_kwargs"]
)

# Decide whether to load or generate
if LOAD_IF_AVAILABLE and DATA_CACHE_PATH.exists() and not FORCE_REGENERATE:
    print(f"Loading dataset from {DATA_CACHE_PATH}")
    with open(DATA_CACHE_PATH, "rb") as f:
        augmented_dataset = pickle.load(f)
else:
    print("Generating dataset...")
    augmented_dataset = builder()

    print(f"Saving dataset to {DATA_CACHE_PATH}")
    with open(DATA_CACHE_PATH, "wb") as f:
        pickle.dump(augmented_dataset, f)



import deeplay as dl

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
    pin_memory=False,  # Load the entire dataset into memory for faster access
)

# Initialize the trainer
trainer = dl.Trainer(max_epochs=30)  # Maximum number of training epochs

trainer.fit(
    clusterer,  # The MIRO model to be trained
    train_loader,  # The DataLoader providing the training data
)