using Random
using SpecialFunctions
using Statistics

"""
    populate_galaxies_julia(h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
                           s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
                           lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s,
                           rsd, Lmin, Lmax)

Populates galaxies based on halo and subhalo data using HOD parameters.

# Arguments
- `h_*`: Arrays containing halo properties
- `s_*`: Arrays containing subhalo properties
- HOD parameters: `lnMcut`, `sigma`, `lnM1`, `kappa`, `alpha`, `alpha_c`, `alpha_s`
- `rsd`: Boolean to apply redshift-space distortions
- `Lmin`, `Lmax`: Box limits for periodic boundary conditions
"""
function populate_galaxies_julia(
    h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
    s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s,
    rsd::Bool, Lmin::Float64, Lmax::Float64
)
    # Pre-allocate results to avoid multiple concatenations
    total_length = length(h_mass) + length(s_mass)
    x_gal = Vector{Float64}(undef, total_length)
    y_gal = Vector{Float64}(undef, total_length)
    z_gal = Vector{Float64}(undef, total_length)
    
    # Calculate parameters once
    Mcut = exp10(lnMcut)
    M1 = exp10(lnM1)
    Lbox = Lmax - Lmin
    sqrt2sigma = sqrt(2) * sigma
    
    # Process central galaxies
    log_ratio_h = log10.(Mcut ./ h_mass)
    p_cen = 0.5 .* erfc.(log_ratio_h ./ sqrt2sigma)
    
    # Use more efficient method to count selected elements
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
    
    # Process satellite galaxies
    log_ratio_s = log10.(Mcut ./ s_mass)
    p_cen_sat = 0.5 .* erfc.(log_ratio_s ./ sqrt2sigma)
    
    # Calculate n_sat more efficiently
    s_count = h_count
    @inbounds for i in 1:length(s_mass)
        mass_diff = s_mass[i] - kappa * Mcut
        if mass_diff > 0
            n_sat_i = (mass_diff / M1)^alpha * p_cen_sat[i]
            prob = n_sat_i / s_n_particles[i]
            
            if rand() < prob
                s_count += 1
                x_gal[s_count] = s_x[i]
                y_gal[s_count] = s_y[i]
                
                if rsd
                    # Apply RSD inline for satellites
                    z_val = s_z[i] + s_host_velocity[i] + alpha_s * (s_velocity[i] - s_host_velocity[i])
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
    
    return x_gal, y_gal, z_gal
end