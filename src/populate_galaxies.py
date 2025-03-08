import numpy as np
import scipy as sp

def populate_galaxies(h, s, HODparams, rsd=True, Lmin=-1000, Lmax=1000):
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
        - lnMcut: Log10 of the minimum mass for a halo to host a central galaxy
        - sigma: Scatter in the minimum mass threshold
        - lnM1: Log10 of the characteristic mass for satellite galaxies
        - kappa: Factor relating central and satellite cutoff masses
        - alpha: Power-law slope of the satellite occupation function
        - alpha_c: Velocity bias parameter for central galaxies (only used when rsd=True)
        - alpha_s: Velocity bias parameter for satellite galaxies (only used when rsd=True)
    rsd : bool, optional
        Whether to apply redshift-space distortion corrections. Default is True.
    Lmin : float, optional
        Minimum coordinate value of the simulation box. Default is -1000.
    Lmax : float, optional
        Maximum coordinate value of the simulation box. Default is 1000.
        
    Returns:
    --------
    dict
        Dictionary containing galaxy positions with keys:
        {'x', 'y', 'z'} where each value is a numpy array of galaxy positions
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
    
    # Copy positions before applying RSD
    zh_rsd = zh.copy()
    zs_rsd = zs.copy()
    
    if rsd:
        # Apply redshift-space distortions using velocity bias parameters
        Lbox = Lmax - Lmin
        zh_rsd += vh + alpha_c * np.random.normal(0, sh, len(Mh))
        zs_rsd += vhost + alpha_s * (vs - vhost)
        
        # Implement periodic boundary conditions using vectorized operations
        zh_rsd = np.mod(zh_rsd - Lmin, Lbox) + Lmin
        zs_rsd = np.mod(zs_rsd - Lmin, Lbox) + Lmin

    # Return dictionary of galaxy positions
    return {
        'x': np.concatenate((xh[Hmask], xs[Smask])),
        'y': np.concatenate((yh[Hmask], ys[Smask])),
        'z': np.concatenate((zh_rsd[Hmask], zs_rsd[Smask]))
    }