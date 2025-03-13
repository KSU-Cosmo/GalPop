import h5py


def save_to_hdf5(data, filename):
    """
    Save a nested dictionary structure containing NumPy arrays to an HDF5 file.

    Parameters:
    -----------
    data : dict
        The nested dictionary structure to save.
        Expected to have 'halo' and 'subsample' as top-level keys.
    filename : str
        Path to the output HDF5 file.
    """
    with h5py.File(filename, "w") as f:
        # Loop through top-level keys (halo, subsample)
        for group_name, group_data in data.items():
            # Create a group for each top-level key
            group = f.create_group(group_name)

            # Loop through arrays in each group
            for key, array in group_data.items():
                # Save each array as a dataset with compression
                group.create_dataset(key, data=array, compression="gzip")


def load_from_hdf5(filename):
    """
    Load a nested dictionary structure containing NumPy arrays from an HDF5 file.

    Parameters:
    -----------
    filename : str
        Path to the HDF5 file to load.

    Returns:
    --------
    dict
        The loaded nested dictionary structure.
    """
    data = {}

    with h5py.File(filename, "r") as f:
        # Loop through top-level groups
        for group_name in f.keys():
            # Create a dictionary for each group
            data[group_name] = {}

            # Loop through datasets in each group
            for key in f[group_name].keys():
                # Load each array
                data[group_name][key] = f[group_name][key][:]

    return data


# Example usage in Python:
"""
# Save data
save_to_hdf5(data, 'cosmology_data.h5')

# Load data
loaded_data = load_from_hdf5('cosmology_data.h5')
"""
