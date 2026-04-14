import h5py

def print_hdf5_structure(name, obj):
    """Recursive function to print all datasets and their shapes."""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}  →  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}  (contains {len(obj)} items)")

# === Inspect each file ===
for fname, label in [("2D_diff.h5", "2D_diff")]:
    print(f"\n=== {label} File Structure ===")
    try:
        with h5py.File(fname, "r") as f:
            # Walk the entire hierarchy
            f.visititems(print_hdf5_structure)
    except Exception as e:
        print(f"Error opening {fname}: {e}")
