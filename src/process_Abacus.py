from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import numpy as np
import glob
from astropy.io import fits
from astropy.table import Table


def process_Abacus_slab(slabname, Mhlow, Mslow, maxsats):
    """
    Process a single Abacus simulation slab file to extract halo and subhalo data.
    
    Parameters
    ----------
    slabname : str
        Path to the Abacus slab file (.asdf format)
    Mhlow : float
        Minimum halo mass threshold in log10(Msun/h)
    Mslow : float
        Minimum subhalo host mass threshold in log10(Msun/h)
    maxsats : int
        Maximum number of satellite subhalos to include per host halo
    
    Returns
    -------
    dict
        Dictionary with two top-level keys ('halo' and 'subsample'), each containing
        a dictionary of arrays with properties for halos and subhalos respectively.
        Arrays within each category have the same length, but halo and subsample
        arrays may have different lengths from each other.
    """
def process_Abacus_slab(slabname, Mhlow, Mslow, maxsats):
    """Process a single Abacus simulation slab file."""
    cat = CompaSOHaloCatalog(
        slabname,
        subsamples=dict(A=True, rv=True),
        fields=['N', 'x_L2com', 'v_L2com', 'npstartA', 'npoutA', 'sigmav3d_L2com'],
        cleaned=True,
    )
    
    # Extract scaling factors from header
    Mpart = cat.header['ParticleMassHMsun']
    inv_velz2kms = 1 / (cat.header['VelZSpace_to_kms'] / cat.header['BoxSizeHMpc'])
    
    # Calculate halo masses and apply mass threshold
    Mh = Mpart * cat.halos["N"]
    Hmask = (Mh > pow(10, Mhlow))
    
    # Create arrays for host properties
    host_masses = np.repeat(Mh, cat.halos["npoutA"])
    host_zvel = np.repeat(cat.halos['v_L2com'][:,2], cat.halos["npoutA"])
    
    # Create combined mask for subhalo selection
    sub_count_mask = np.concatenate([np.concatenate((np.ones(min(n, maxsats)), 
                     np.zeros(max(n - maxsats, 0)))) for n in cat.halos["npoutA"]])
    Smask = np.logical_and(host_masses > pow(10, Mslow), sub_count_mask.astype(bool))
    
    # Extract and scale halo properties
    halo_props = {
        'mass': Mh[Hmask].value,
        'x': cat.halos['x_L2com'][Hmask, 0].value,
        'y': cat.halos['x_L2com'][Hmask, 1].value,
        'z': cat.halos['x_L2com'][Hmask, 2].value,
        'velocity': cat.halos['v_L2com'][Hmask, 2].value * inv_velz2kms,
        'sigma': np.sqrt(cat.halos['sigmav3d_L2com'][Hmask]).value * inv_velz2kms
    }
    
    # Extract and scale subsample properties
    subsample_props = {
        'mass': host_masses[Smask].value,
        'host_velocity': host_zvel[Smask].value * inv_velz2kms,
        'n_particles': np.repeat(cat.halos["npoutA"], cat.halos["npoutA"])[Smask].value,
        'x': cat.subsamples['pos'][Smask, 0].value,
        'y': cat.subsamples['pos'][Smask, 1].value,
        'z': cat.subsamples['pos'][Smask, 2].value,
        'velocity': cat.subsamples['vel'][Smask, 2].value * inv_velz2kms
    }
    
    return {'halo': halo_props, 'subsample': subsample_props}


def process_Abacus_directory(dir_path, Mhlow, Mslow, maxsats):
    """
    Process multiple Abacus simulation slab files from a directory.
    
    Parameters
    ----------
    dir_path : str
        Directory containing Abacus slab files (should end with '/')
    Mhlow : float
        Minimum halo mass threshold in log10(Msun/h)
    Mslow : float
        Minimum subhalo host mass threshold in log10(Msun/h)
    maxsats : int
        Maximum number of satellite subhalos to include per host halo
        
    Returns
    -------
    dict
        Combined dictionary with halo and subsample data from all processed slabs
    """
    slablist = glob.glob(f"{dir_path}*.asdf")
    
    # Initialize the result dictionary with empty lists
    result = {
        'halo': {
            'mass': [], 'x': [], 'y': [], 'z': [], 'sigma': [], 'velocity': []
        },
        'subsample': {
            'mass': [], 'host_velocity': [], 'n_particles': [],
            'x': [], 'y': [], 'z': [], 'velocity': []
        }
    }
    
    # Process slabs
    for slab in slablist:
        print(f"Processing slab: {slab}")
        try:
            slab_result = process_Abacus_slab(slab, Mhlow, Mslow, maxsats)
            
            # Append data from this slab to our results
            for category in result:
                for key in result[category]:
                    result[category][key] = np.append(
                        result[category][key], 
                        slab_result[category][key]
                    )
        except Exception as e:
            print(f"Error processing slab {slab}: {e}")
    
    return result


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


def main():
    """
    Example use of the Abacus processing functions.
    
    This function demonstrates how to process Abacus data and save the results.
    Uncomment and modify the parameters to use.
    """
    # Example usage
    # dir_path = "/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_base_c000_ph000/halos/z0.800/halo_info/"
    # Mhlow = 12.5       # Minimum log10 halo mass
    # Mslow = 13.5       # Minimum log10 subhalo host mass
    # maxsats = 25       # Maximum satellites per host
    # 
    # results = process_Abacus_directory(dir_path, Mhlow, Mslow, maxsats)
    # save_results_fits(results, "abacus_results.fits")
    # 
    # # Later, to read the data:
    # halo_data, subsample_data = read_results_fits("abacus_results.fits")
    pass


if __name__ == "__main__":
    main()
