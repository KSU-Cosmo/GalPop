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
    # Load the catalog data
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
    
    # Extract and scale halo properties
    halo_props = {
        'mass': Mh[Hmask].value,
        'x': cat.halos['x_L2com'][Hmask, 0].value,
        'y': cat.halos['x_L2com'][Hmask, 1].value,
        'z': cat.halos['x_L2com'][Hmask, 2].value,
        'velocity': cat.halos['v_L2com'][Hmask, 2].value * inv_velz2kms,
        'sigma': np.sqrt(cat.halos['sigmav3d_L2com'][Hmask]).value * inv_velz2kms
    }
    
    # Vectorized approach for creating sub_count_mask
    n_particles = cat.halos["npoutA"]
    mask_lengths = np.minimum(n_particles, maxsats)
    
    # Calculate starting indices for each halo's satellites
    start_indices = np.zeros(len(n_particles), dtype=int)
    if len(n_particles) > 1:
        start_indices[1:] = np.cumsum(n_particles[:-1])
    
    # Create the mask more efficiently
    total_length = np.sum(n_particles)
    sub_count_mask = np.zeros(total_length, dtype=bool)
    
    for i, (start, length) in enumerate(zip(start_indices, mask_lengths)):
        sub_count_mask[start:start+length] = True
    
    # Create host property arrays more efficiently
    # Only create for hosts that pass the mass threshold for subsamples
    host_mass_filter = Mh > pow(10, Mslow)
    
    # Filter and repeat only the necessary host properties
    valid_hosts_indices = np.where(host_mass_filter)[0]
    valid_n_particles = n_particles[valid_hosts_indices]
    
    # Generate arrays only for valid hosts
    host_masses = np.repeat(Mh[valid_hosts_indices], valid_n_particles)
    host_zvel = np.repeat(cat.halos['v_L2com'][valid_hosts_indices, 2], valid_n_particles)
    
    # Calculate cumulative particle counts for indexing
    cumul_particles = np.zeros(len(n_particles) + 1, dtype=int)
    cumul_particles[1:] = np.cumsum(n_particles)
    
    # Create mask for all particles (from hosts passing threshold)
    all_particles_mask = np.zeros(total_length, dtype=bool)
    for i in valid_hosts_indices:
        all_particles_mask[cumul_particles[i]:cumul_particles[i+1]] = True
    
    # Final mask is the intersection of all_particles_mask and sub_count_mask
    Smask = np.logical_and(all_particles_mask, sub_count_mask)
    
    # For each valid host, get the larger of n_particles and maxsats
    # We'll create an array to map each subsample back to its host index
    host_idx_map = np.zeros(total_length, dtype=int)
    for i in range(len(n_particles)):
        start = cumul_particles[i]
        end = cumul_particles[i+1]
        host_idx_map[start:end] = i
    
    # Get host indices for selected subsamples
    selected_host_indices = host_idx_map[Smask]
    
    # Get the larger of n_particles and maxsats for each selected host
    smallest_n_particles = np.minimum(n_particles[selected_host_indices], maxsats)
    
    # Extract and scale subsample properties
    subsample_props = {
        'mass': host_masses[sub_count_mask[all_particles_mask]].value,
        'host_velocity': host_zvel[sub_count_mask[all_particles_mask]].value * inv_velz2kms,
        'n_particles': smallest_n_particles.value,  # Use the larger of n_particles and maxsats
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
    total_slabs = len(slablist)
    
    # Initialize temporary storage for collecting arrays before concatenation
    temp_results = {
        'halo': {
            'mass': [], 'x': [], 'y': [], 'z': [], 'sigma': [], 'velocity': []
        },
        'subsample': {
            'mass': [], 'host_velocity': [], 'n_particles': [],
            'x': [], 'y': [], 'z': [], 'velocity': []
        }
    }
    
    # Track successfully processed slabs and failures
    successful_slabs = 0
    failed_slabs = []
    
    # Process slabs
    for i, slab in enumerate(slablist):
        print(f"Processing slab {i+1}/{total_slabs}: {slab}")
        
        try:
            slab_result = process_Abacus_slab(slab, Mhlow, Mslow, maxsats)
            
            # Append data from this slab to our temporary results
            for category in temp_results:
                for key in temp_results[category]:
                    temp_results[category][key].append(slab_result[category][key])
            
            successful_slabs += 1
            
            # Print progress
            print(f"  - Halos found: {len(slab_result['halo']['mass'])}")
            print(f"  - Subsamples found: {len(slab_result['subsample']['mass'])}")
            print(f"  - Progress: {successful_slabs}/{total_slabs} slabs processed successfully")
            
        except Exception as e:
            failed_slabs.append((slab, str(e)))
            print(f"ERROR processing slab {slab}:")
            print(f"  - {e}")
    
    # Consolidate results - concatenate arrays once at the end
    result = {}
    for category in temp_results:
        result[category] = {}
        for key in temp_results[category]:
            if temp_results[category][key]:  # Only concatenate if there are arrays to combine
                result[category][key] = np.concatenate(temp_results[category][key])
            else:
                result[category][key] = np.array([])
    
    # Report on processing results
    print("\nProcessing Summary:")
    print(f"  - Total slabs: {total_slabs}")
    print(f"  - Successfully processed: {successful_slabs}")
    print(f"  - Failed: {len(failed_slabs)}")
    
    if failed_slabs:
        print("\nFailed slabs:")
        for slab, error in failed_slabs:
            print(f"  - {slab}: {error}")
    
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


if __name__ == "__main__":
    main()