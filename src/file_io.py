from astropy.io import fits
from astropy.table import Table
import numpy as np


def save_results_fits(results, filename):
    """
    Save results dictionary to a FITS file with multiple extensions.

    Parameters
    ----------
    results : dict
        Dictionary with halo and subsample data as returned by 
        process_Abacus_directory
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
    h_mass = np.asarray(h['mass'], dtype=np.float32)
    h_x = np.asarray(h['x'], dtype=np.float32)
    h_y = np.asarray(h['y'], dtype=np.float32)
    h_z = np.asarray(h['z'], dtype=np.float32)
    h_velocity = np.asarray(h['velocity'], dtype=np.float32)
    h_sigma = np.asarray(h['sigma'], dtype=np.float32)

    # Extract subsample arrays, ensuring float64 type
    s_mass = np.asarray(s['mass'], dtype=np.float32)
    s_host_velocity = np.asarray(s['host_velocity'], dtype=np.float32)
    s_n_particles = np.asarray(s['n_particles'], dtype=np.int32)
    s_x = np.asarray(s['x'], dtype=np.float32)
    s_y = np.asarray(s['y'], dtype=np.float32)
    s_z = np.asarray(s['z'], dtype=np.float32)
    s_velocity = np.asarray(s['velocity'], dtype=np.float32)

    return (
        h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
        s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity
    )
