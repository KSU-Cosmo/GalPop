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
