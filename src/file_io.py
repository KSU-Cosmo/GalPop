from astropy.io import fits
from astropy.table import Table

def save_results_fits(results, filename):
    """
    Save results dictionary to a FITS file with multiple extensions.
    
    Parameters
    ----------
    results : dict
        Dictionary with halo and subsample data as returned by process_Abacus_directory
    filename : str
        Output FITS filename
        
    Notes
    -----
    Creates a FITS file with separate table extensions for halos and subsamples
    """
    hdul = fits.HDUList([fits.PrimaryHDU()])
    
    # Add each category as a separate extension
    for category in results:
        # Create a Table from the arrays
        t = Table()
        for key in results[category]:
            t[key] = results[category][key]
        
        # Add table as a binary table extension
        hdul.append(fits.BinTableHDU(t, name=category))
    
    hdul.writeto(filename, overwrite=True)
    print(f"Results saved to {filename}")
    
    # Print summary of saved data
    print("\nSaved Data Summary:")
    for category in results:
        if results[category]:
            array_length = len(next(iter(results[category].values())))
            print(f"  - {category.capitalize()}: {array_length} entries")


def read_results_fits(filename):
    """
    Read halo and subsample data from a previously saved FITS file.
    
    Parameters
    ----------
    filename : str
        Path to the FITS file to read
        
    Returns
    -------
    tuple
        (halo_data, subsample_data) - two astropy Table objects containing
        the halo and subsample data
    """
    hdul = fits.open(filename)

    # Access halo data
    halo_data = hdul['halo'].data
    print(f"Number of halos: {len(halo_data)}")

    # Access subsample data
    subsample_data = hdul['subsample'].data
    print(f"Number of subsamples: {len(subsample_data)}")
    
    return halo_data, subsample_data

def extract_arrays(h, s):
    """
    Extract individual arrays from halo and subsample dictionaries.
    
    Parameters:
    -----------
    h : dict
        Dictionary containing halo data with keys:
        {'mass', 'x', 'y', 'z', 'sigma', 'velocity'}
    s : dict
        Dictionary containing subsample data with keys:
        {'mass', 'host_velocity', 'n_particles', 'x', 'y', 'z', 'velocity'}
    
    Returns:
    --------
    tuple
        Tuple containing all extracted arrays in the order:
        (h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
         s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity)
    """
    # Extract halo arrays
    h_mass = h['mass']
    h_x = h['x']
    h_y = h['y']
    h_z = h['z']
    h_velocity = h['velocity']
    h_sigma = h['sigma']
    
    # Extract subsample arrays
    s_mass = s['mass']
    s_host_velocity = s['host_velocity']
    s_n_particles = s['n_particles']
    s_x = s['x']
    s_y = s['y']
    s_z = s['z']
    s_velocity = s['velocity']
    
    return (
        h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
        s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity
    )