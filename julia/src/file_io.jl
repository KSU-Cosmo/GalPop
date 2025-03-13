using HDF5

function save_to_hdf5(data::Dict, filename::String)
    """
    Save a nested dictionary structure containing arrays to an HDF5 file.
    
    Parameters:
    -----------
    data : Dict
        The nested dictionary structure to save.
        Expected to have 'halo' and 'subsample' as top-level keys.
    filename : String
        Path to the output HDF5 file.
    """
    h5open(filename, "w") do file
        # Loop through top-level keys (halo, subsample)
        for (group_name, group_data) in data
            # Create a group for each top-level key
            group = create_group(file, group_name)
            
            # Loop through arrays in each group
            for (key, array) in group_data
                # Save each array as a dataset with compression
                group[key] = array
                # Apply compression
                set_dims_h5(group[key], size(array)...; chunks=(min(1000, size(array,1)), ))
            end
        end
    end
end

function load_from_hdf5(filename::String)
    """
    Load a nested dictionary structure containing arrays from an HDF5 file.
    
    Parameters:
    -----------
    filename : String
        Path to the HDF5 file to load.
        
    Returns:
    --------
    Dict
        The loaded nested dictionary structure.
    """
    data = Dict()
    
    h5open(filename, "r") do file
        # Loop through top-level groups
        for group_name in keys(file)
            # Create a dictionary for each group
            data[group_name] = Dict()
            
            # Loop through datasets in each group
            for key in keys(file[group_name])
                # Load each array
                data[group_name][key] = read(file[group_name][key])
            end
        end
    end
    
    return data
end

# Example usage in Julia:
# Save data
# save_to_hdf5(data, \"cosmology_data.h5\")

# Load data
# loaded_data = load_from_hdf5(\"cosmology_data.h5\")