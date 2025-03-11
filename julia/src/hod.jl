# File: julia/src/hod.jl

"""
    calculate_p_cen(halo_mass, Mcut, sigma)

Calculate the probability that a halo hosts a central galaxy.

# Arguments
- `halo_mass::Vector{Float32}`: Halo masses
- `Mcut::Float64`: Minimum mass threshold
- `sigma::Float64`: Scatter in the HOD

# Returns
- Vector{Float64}: Probability values for each halo
"""
function calculate_p_cen(halo_mass, Mcut, sigma)
    sqrt2sigma = sqrt(2) * sigma
    log_ratio = log10.(Mcut ./ halo_mass)
    return 0.5 .* erfc.(log_ratio ./ sqrt2sigma)
end

"""
    calculate_n_sat(subhalo_mass, Mcut, M1, alpha, kappa, p_cen_sat)

Calculate the expected number of satellite galaxies per subhalo.

# Arguments
- `subhalo_mass::Float32`: Subhalo mass
- `Mcut::Float64`: Minimum mass threshold
- `M1::Float64`: Characteristic satellite mass
- `alpha::Float64`: Satellite slope
- `kappa::Float64`: Satellite threshold mass relative to Mcut
- `p_cen_sat::Float64`: Central galaxy probability

# Returns
- Float64: Expected number of satellite galaxies
"""
function calculate_n_sat(subhalo_mass, Mcut, M1, alpha, kappa, p_cen_sat)
    mass_diff = subhalo_mass - kappa * Mcut
    if mass_diff > 0
        return (mass_diff / M1)^alpha * p_cen_sat
    else
        return 0.0
    end
end

"""
    populate_galaxies(halos, subhalos, hod_params)

Populates galaxies based on halo and subhalo data using HOD parameters.

# Arguments
- `halos`: NamedTuple containing:
  - mass::Vector{Float32}: Halo masses
  - x, y, z::Vector{Float32}: Halo positions
  - velocity::Vector{Float32}: Halo velocities
  - sigma::Vector{Float32}: Halo velocity dispersions
  
- `subhalos`: NamedTuple containing:
  - mass::Vector{Float32}: Subhalo masses
  - host_velocity::Vector{Float32}: Host halo velocities
  - n_particles::Vector{Int32}: Number of particles in subhalo
  - x, y, z::Vector{Float32}: Subhalo positions
  - velocity::Vector{Float32}: Subhalo velocities
  
- `hod_params`: NamedTuple containing HOD parameters:
  - lnMcut: Log10 of minimum halo mass for central galaxies
  - sigma: Scatter in the central galaxy HOD
  - lnM1: Log10 of characteristic mass for satellite galaxies
  - kappa: Satellite threshold mass relative to Mcut
  - alpha: Satellite slope
  - alpha_c: Central galaxy velocity bias
  - alpha_s: Satellite galaxy velocity bias
  - rsd: Boolean to apply redshift-space distortions
  - Lmin, Lmax: Box limits for periodic boundary conditions

# Returns
- NamedTuple with (x, y, z, count) containing galaxy positions and total count
"""
function populate_galaxies(halos, subhalos, hod_params)
    # Extract arrays from the NamedTuples
    h_mass = halos.mass
    h_x = halos.x
    h_y = halos.y
    h_z = halos.z
    h_velocity = halos.velocity
    h_sigma = halos.sigma

    s_mass = subhalos.mass
    s_host_velocity = subhalos.host_velocity
    s_n_particles = subhalos.n_particles
    s_x = subhalos.x
    s_y = subhalos.y
    s_z = subhalos.z
    s_velocity = subhalos.velocity

    lnMcut = hod_params.lnMcut
    sigma = hod_params.sigma
    lnM1 = hod_params.lnM1
    kappa = hod_params.kappa
    alpha = hod_params.alpha
    alpha_c = hod_params.alpha_c
    alpha_s = hod_params.alpha_s
    rsd = hod_params.rsd
    Lmin = hod_params.Lmin
    Lmax = hod_params.Lmax

    # Pre-allocate results to avoid multiple concatenations
    total_length = length(h_mass) + length(s_mass)
    x_gal = Vector{Float64}(undef, total_length)
    y_gal = Vector{Float64}(undef, total_length)
    z_gal = Vector{Float64}(undef, total_length)

    # Calculate parameters once
    Mcut = exp10(lnMcut)
    M1 = exp10(lnM1)
    Lbox = Lmax - Lmin

    # Calculate central galaxy probabilities
    p_cen = calculate_p_cen(h_mass, Mcut, sigma)

    # Process central galaxies
    h_count = 0
    @inbounds for i in 1:length(p_cen)
        if rand() < p_cen[i]
            h_count += 1
            x_gal[h_count] = h_x[i]
            y_gal[h_count] = h_y[i]

            if rsd
                # Apply RSD inline for centrals
                z_val = h_z[i] + h_velocity[i] + alpha_c * randn() * h_sigma[i]
                # Apply periodic boundary inline
                z_gal[h_count] = mod(z_val - Lmin, Lbox) + Lmin
            else
                z_gal[h_count] = h_z[i]
            end
        end
    end

    # Calculate subhalo central probabilities
    p_cen_sat = calculate_p_cen(s_mass, Mcut, sigma)

    # Process satellite galaxies
    s_count = h_count
    @inbounds for i in 1:length(s_mass)
        # Calculate expected number of satellites
        n_sat_i = calculate_n_sat(s_mass[i], Mcut, M1, alpha, kappa, p_cen_sat[i])

        # Normalize by number of particles and check probability
        if n_sat_i > 0
            prob = n_sat_i / s_n_particles[i]

            if rand() < prob
                s_count += 1
                x_gal[s_count] = s_x[i]
                y_gal[s_count] = s_y[i]

                if rsd
                    # Apply RSD inline for satellites
                    z_val =
                        s_z[i] +
                        s_host_velocity[i] +
                        alpha_s * (s_velocity[i] - s_host_velocity[i])
                    # Apply periodic boundary inline
                    z_gal[s_count] = mod(z_val - Lmin, Lbox) + Lmin
                else
                    z_gal[s_count] = s_z[i]
                end
            end
        end
    end

    # Resize arrays to actual count
    resize!(x_gal, s_count)
    resize!(y_gal, s_count)
    resize!(z_gal, s_count)

    return (x=x_gal, y=y_gal, z=z_gal, count=s_count)
end
