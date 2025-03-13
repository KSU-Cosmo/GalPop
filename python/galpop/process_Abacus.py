from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import numpy as np
import glob


def process_Abacus_slab(slabname, Mhlow, Mslow, maxsats):
    """
    Process a single Abacus simulation slab file to extract halo and subhalo
    data.

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
        Dictionary with two top-level keys ('halo' and 'subsample'), each
        containing a dictionary of arrays with properties for halos and
        subhalos respectively. Arrays within each category have the same
        length, but halo and subsample arrays may have different lengths from
        each other.
    """
    # Load the catalog data
    cat = CompaSOHaloCatalog(
        slabname,
        subsamples=dict(A=True, rv=True),
        fields=["N", "x_L2com", "v_L2com", "npstartA", "npoutA", "sigmav3d_L2com"],
        cleaned=True,
    )

    # Extract scaling factors from header
    Mpart = cat.header["ParticleMassHMsun"]
    inv_velz2kms = 1 / cat.header["VelZSpace_to_kms"]
    inv_velz2kms /= cat.header["BoxSizeHMpc"]

    # Calculate halo masses and apply mass threshold
    Mh = Mpart * cat.halos["N"]
    Hmask = Mh > pow(10, Mhlow)

    # Extract and scale halo properties
    halo_props = {
        "mass": Mh[Hmask].value,
        "x": cat.halos["x_L2com"][Hmask, 0].value,
        "y": cat.halos["x_L2com"][Hmask, 1].value,
        "z": cat.halos["x_L2com"][Hmask, 2].value,
        "velocity": cat.halos["v_L2com"][Hmask, 2].value * inv_velz2kms,
        "sigma": np.sqrt(cat.halos["sigmav3d_L2com"][Hmask]).value * inv_velz2kms,
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
        sub_count_mask[start : start + length] = True

    # Create host property arrays more efficiently
    # Only create for hosts that pass the mass threshold for subsamples
    host_mass_filter = Mh > pow(10, Mslow)

    # Filter and repeat only the necessary host properties
    valid_hosts_indices = np.where(host_mass_filter)[0]
    valid_n_particles = n_particles[valid_hosts_indices]

    # Generate arrays only for valid hosts
    host_masses = np.repeat(Mh[valid_hosts_indices], valid_n_particles)
    host_zvel = np.repeat(cat.halos["v_L2com"][valid_hosts_indices, 2], valid_n_particles)

    # Calculate cumulative particle counts for indexing
    cumul_particles = np.zeros(len(n_particles) + 1, dtype=int)
    cumul_particles[1:] = np.cumsum(n_particles)

    # Create mask for all particles (from hosts passing threshold)
    all_particles_mask = np.zeros(total_length, dtype=bool)
    for i in valid_hosts_indices:
        all_particles_mask[cumul_particles[i] : cumul_particles[i + 1]] = True

    # Final mask is the intersection of all_particles_mask and sub_count_mask
    Smask = np.logical_and(all_particles_mask, sub_count_mask)

    # For each valid host, get the larger of n_particles and maxsats
    # We'll create an array to map each subsample back to its host index
    host_idx_map = np.zeros(total_length, dtype=int)
    for i in range(len(n_particles)):
        start = cumul_particles[i]
        end = cumul_particles[i + 1]
        host_idx_map[start:end] = i

    # Get host indices for selected subsamples
    selected_host_indices = host_idx_map[Smask]

    # Get the larger of n_particles and maxsats for each selected host
    smallest_n_particles = np.minimum(n_particles[selected_host_indices], maxsats)

    # Extract and scale subsample properties
    subsample_props = {
        "mass": host_masses[sub_count_mask[all_particles_mask]].value,
        "host_velocity": host_zvel[sub_count_mask[all_particles_mask]].value * inv_velz2kms,
        # Use the larger of n_particles and maxsats
        "n_particles": smallest_n_particles.value,
        "x": cat.subsamples["pos"][Smask, 0].value,
        "y": cat.subsamples["pos"][Smask, 1].value,
        "z": cat.subsamples["pos"][Smask, 2].value,
        "velocity": cat.subsamples["vel"][Smask, 2].value * inv_velz2kms,
    }

    return {"halo": halo_props, "subsample": subsample_props}


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

    # Initialize temporary storage for collecting arrays before concatenation
    temp_results = {
        "halo": {"mass": [], "x": [], "y": [], "z": [], "sigma": [], "velocity": []},
        "subsample": {
            "mass": [],
            "host_velocity": [],
            "n_particles": [],
            "x": [],
            "y": [],
            "z": [],
            "velocity": [],
        },
    }

    # Process each slab
    for i, slab in enumerate(slablist):
        print(f"Processing slab {i+1}/{len(slablist)}: {slab}")
        slab_result = process_Abacus_slab(slab, Mhlow, Mslow, maxsats)

        # Append data from this slab
        for category in temp_results:
            for key in temp_results[category]:
                temp_results[category][key].append(slab_result[category][key])

    # Consolidate results - concatenate arrays
    result = {}
    for category in temp_results:
        result[category] = {}
        for key in temp_results[category]:
            if temp_results[category][key]:
                result[category][key] = np.concatenate(temp_results[category][key])
            else:
                result[category][key] = np.array([])

    return result
