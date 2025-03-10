using FITSIO

"""
    read_results_fits(filename)

Read halo and subsample data from a previously saved FITS file.

# Parameters
- `filename::String`: Path to the FITS file to read

# Returns
- `Tuple{FITS.TableHDU, FITS.TableHDU}`: Two FITS table objects containing
  the halo and subsample data
"""
function read_results_fits(filename)
    # Open the FITS file
    fits = FITS(filename)

    # Access halo data
    halo_data = fits["halo"]
    
    # Get number of rows directly from header
    halo_nrows = FITSIO.read_header(halo_data)["NAXIS2"]
    println("Number of halos: $halo_nrows")

    # Access subsample data
    subsample_data = fits["subsample"]
    
    # Get number of rows directly from header
    subsample_nrows = FITSIO.read_header(subsample_data)["NAXIS2"]
    println("Number of subsamples: $subsample_nrows")
    
    return halo_data, subsample_data
end

"""
    extract_arrays_from_fits(h, s)

Extract individual arrays from halo and subsample FITS TableHDU objects.

# Parameters
- `h::FITS.TableHDU`: Halo data table from FITS file
- `s::FITS.TableHDU`: Subsample data table from FITS file

# Returns
- Tuple containing all extracted arrays in the order:
  (h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
   s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity)
"""
function extract_arrays_from_fits(h, s)
    # Print column names to verify what's available
    println("Halo columns: ", FITSIO.colnames(h))
    println("Subsample columns: ", FITSIO.colnames(s))
    
    # Extract halo arrays
    h_mass = read(h, "mass")
    h_x = read(h, "x")
    h_y = read(h, "y")
    h_z = read(h, "z")
    h_velocity = read(h, "velocity")
    h_sigma = read(h, "sigma")
    
    # Extract subsample arrays
    s_mass = read(s, "mass")
    s_host_velocity = read(s, "host_velocity")
    s_n_particles = read(s, "n_particles")
    s_x = read(s, "x")
    s_y = read(s, "y")
    s_z = read(s, "z")
    s_velocity = read(s, "velocity")
    
    return (
        h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
        s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity
    )
end