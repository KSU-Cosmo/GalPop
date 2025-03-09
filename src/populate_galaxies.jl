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
    # Calculate parameters
    Mcut = 10.0^lnMcut
    M1 = 10.0^lnM1
    Lbox = Lmax - Lmin
    
    # Probability of central galaxies
    p_cen = 0.5 .* erfc.((log10.(Mcut ./ h_mass)) ./ (sqrt(2) * sigma))
    p_cen_sat = 0.5 .* erfc.((log10.(Mcut ./ s_mass)) ./ (sqrt(2) * sigma))
    
    # Number of satellite galaxies
    n_sat = ((s_mass .- kappa*Mcut) ./ M1)
    n_sat[n_sat .< 0] .= 0
    n_sat = n_sat.^alpha .* p_cen_sat
    
    # Select central galaxies using random sampling
    Hmask = rand(length(p_cen)) .< p_cen
    
    # Select satellite galaxies using random sampling
    Smask = rand(length(n_sat)) .< n_sat ./ s_n_particles
    
    if rsd
        # Apply redshift-space distortions
        h_z .+= h_velocity .+ alpha_c .* randn(length(h_mass)) .* h_sigma
        h_s .+= s_host_velocity .+ alpha_s .* (s_velocity .- s_host_velocity)
        
        # Apply periodic boundary conditions
        h_z.= mod.(h_z .- Lmin, Lbox) .+ Lmin
        h_s .= mod.(h_s .- Lmin, Lbox) .+ Lmin
    end
    
    # Select galaxies using the masks
    x_gal = vcat(h_x[Hmask], s_x[Smask])
    y_gal = vcat(h_y[Hmask], s_y[Smask])
    z_gal = vcat(zh[Hmask], zs[Smask])
    
    return x_gal, y_gal, z_gal
end