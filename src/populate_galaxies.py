import numpy as np
import scipy as sp

def populate_galaxies(h, s, HODparams, rsd=True):
    """
    Generate galaxies based on halo and subsample data using Halo Occupation Distribution (HOD) parameters.
    
    Parameters:
    -----------
    h : dict
        Dictionary containing halo data with keys:
        {'mass', 'x', 'y', 'z', 'sigma', 'velocity'}
    s : dict
        Dictionary containing subsample data with keys:
        {'mass', 'host_velocity', 'n_particles', 'x', 'y', 'z', 'velocity'}
    HODparams : list or tuple
        List of HOD parameters [lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s].
        The default values for velocity bias are alpha_c = 0, alpha_s = 1.
    rsd : bool
        True by default. Whether or not to apply the rsd correction.
    Returns:
    --------
    tuple of numpy arrays
        (x_galaxies, y_galaxies, z_galaxies)
    """
    # Unpack halo data and ensure float64 type
    Mh = np.asarray(h['mass'], dtype=np.float64)
    xh = np.asarray(h['x'], dtype=np.float64)
    yh = np.asarray(h['y'], dtype=np.float64)
    zh = np.asarray(h['z'], dtype=np.float64)
    vh = np.asarray(h['velocity'], dtype=np.float64)
    sh = np.asarray(h['sigma'], dtype=np.float64)
    
    # Unpack subsample data and ensure float64 type
    Ms = np.asarray(s['mass'], dtype=np.float64)
    vhost = np.asarray(s['host_velocity'], dtype=np.float64)
    ns = np.asarray(s['n_particles'], dtype=np.float64)
    xs = np.asarray(s['x'], dtype=np.float64)
    ys = np.asarray(s['y'], dtype=np.float64)
    zs = np.asarray(s['z'], dtype=np.float64)
    vs = np.asarray(s['velocity'], dtype=np.float64)
    
    # Unpack HOD parameters
    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s = HODparams
    
    # Calculate parameters
    Mcut = 10.0**lnMcut
    M1 = 10.0**lnM1
    
    # Probability of central galaxies
    p_cen = 0.5 * sp.special.erfc((np.log10(Mcut/Mh)) / np.sqrt(2) / sigma)
    p_cen_sat = 0.5 * sp.special.erfc((np.log10(Mcut/Ms)) / np.sqrt(2) / sigma)
    
    # Number of satellite galaxies
    n_sat = ((Ms - kappa*Mcut) / M1)
    n_sat[n_sat < 0] = 0
    n_sat = n_sat**alpha * p_cen_sat
    
    # Select central galaxies
    random_value = np.random.rand(*p_cen.shape)
    Hmask = random_value < p_cen
    
    # Select satellite galaxies
    random_value = np.random.rand(*n_sat.shape)
    Smask = random_value < n_sat/ns
    
    if rsd:
        zh += vh + alpha_c * np.random.normal(0, sh, len(Mh))
        zs += vhost + alpha_s * (vs - vhost)

    # Return concatenated coordinates of central and satellite galaxies
    return (
        np.concatenate((xh[Hmask], xs[Smask])),
        np.concatenate((yh[Hmask], ys[Smask])),
        np.concatenate((zh[Hmask], zs[Smask]))
    )