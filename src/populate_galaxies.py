import numpy as np
import scipy as sp

def generate_galaxies(data_dict, HODparams):
    """
    Generate galaxies based on halo and subsample data using Halo Occupation Distribution (HOD) parameters.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing halo and subsample data with keys:
        - 'halo': {'mass', 'x', 'y', 'z', 'sigma', 'velocity'}
        - 'subsample': {'mass', 'host_velocity', 'n_particles', 'x', 'y', 'z', 'velocity'}
    HODparams : list or tuple
        List of HOD parameters [lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s]
    
    Returns:
    --------
    tuple of numpy arrays
        (x_galaxies, y_galaxies, z_galaxies)
    """
    # Unpack halo data and ensure float64 type
    Mh = np.asarray(data_dict['halo']['mass'], dtype=np.float64)
    xh = np.asarray(data_dict['halo']['x'], dtype=np.float64)
    yh = np.asarray(data_dict['halo']['y'], dtype=np.float64)
    zh = np.asarray(data_dict['halo']['z'], dtype=np.float64)
    sh = np.asarray(data_dict['halo']['sigma'], dtype=np.float64)
    
    # Unpack subsample data and ensure float64 type
    Ms = np.asarray(data_dict['subsample']['mass'], dtype=np.float64)
    vhost = np.asarray(data_dict['subsample']['host_velocity'], dtype=np.float64)
    ns = np.asarray(data_dict['subsample']['n_particles'], dtype=np.float64)
    xs = np.asarray(data_dict['subsample']['x'], dtype=np.float64)
    ys = np.asarray(data_dict['subsample']['y'], dtype=np.float64)
    zs = np.asarray(data_dict['subsample']['z'], dtype=np.float64)
    vs = np.asarray(data_dict['subsample']['velocity'], dtype=np.float64)
    
    # Unpack HOD parameters
    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s = HODparams
    
    # Calculate parameters
    Mcut = 10.0**lnMcut
    M1 = 10.0**lnM1
    
    # Probability of central galaxies
    p_cen = 0.5 * sp.special.erfc((np.log10(Mcut/Mh)) / np.sqrt(2) / sigma)
    p_cen_sat = 0.5 * sp.special.erfc((np.log10(Mcut/Ms)) / np.sqrt(2) / sigma)
    
    # Number of satellite galaxies
    n_sat = ((Ms - Mcut) / M1)
    n_sat[n_sat < 0] = 0
    n_sat = n_sat**alpha * p_cen_sat
    
    # Add scatter to z-coordinate of central galaxies
    zh += alpha_c * np.random.normal(0, sh, len(Mh))
    
    # Select central galaxies
    random_value = np.random.rand(*p_cen.shape)
    Hmask = random_value < p_cen
    
    # Adjust satellite galaxy velocities
    zs += vhost + alpha_s * (zs - vhost)
    
    # Select satellite galaxies
    random_value = np.random.rand(*n_sat.shape)
    Smask = random_value < n_sat/ns
    
    # Return concatenated coordinates of central and satellite galaxies
    return (
        np.concatenate((xh[Hmask], xs[Smask])),
        np.concatenate((yh[Hmask], ys[Smask])),
        np.concatenate((zh[Hmask], zs[Smask]))
    )