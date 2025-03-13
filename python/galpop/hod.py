import os
import numpy as np
from julia import Julia
from julia import Main
from typing import Dict, Union


class GalPopWrapper:
    """
    Performance-optimized Python wrapper for the HOD function in GalPop.
    """

    def __init__(self, julia_project_path=None, precompile=True):
        """
        Initialize the wrapper by setting up Julia environment.

        Parameters
        ----------
        julia_project_path : str, optional
            Path to the Julia project directory. If None, will try to find it
            automatically.
        precompile : bool
            Whether to precompile the function for better performance.
        """
        # Initialize Julia with shared memory for arrays when possible
        jl = Julia(compiled_modules=False)

        # Set up the project path
        if julia_project_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            julia_project_path = os.path.abspath(
                os.path.join(current_dir, "..", "..", "..", "julia")
            )

        # Check if the path exists
        hod_file_path = os.path.join(julia_project_path, "src", "hod.jl")
        if not os.path.exists(hod_file_path):
            raise FileNotFoundError(
                f"Could not find src/hod.jl in {julia_project_path}"
            )

        # Activate the Julia project
        Main.eval(f'using Pkg; Pkg.activate("{julia_project_path}")')
        Main.eval(f'include("{hod_file_path}")')

        # Try to determine module structure
        try:
            Main.eval("using .HOD")
            self.module_name = "HOD"
        except:
            self.module_name = "Main"

        # Create a performance-optimized wrapper function in Julia
        # This avoids recreating NamedTuples for every call
        Main.eval(
            f"""
        function fast_populate_galaxies(
            h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
            s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
            lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s, rsd, Lmin,
            Lmax
        )
            # Create NamedTuples only once
            halos = (
                mass=h_mass, 
                x=h_x, 
                y=h_y, 
                z=h_z, 
                velocity=h_velocity, 
                sigma=h_sigma
            )

            subhalos = (
                mass=s_mass,
                host_velocity=s_host_velocity,
                n_particles=s_n_particles,
                x=s_x,
                y=s_y,
                z=s_z,
                velocity=s_velocity
            )

            hod_params = (
                lnMcut=lnMcut,
                sigma=sigma,
                lnM1=lnM1,
                kappa=kappa,
                alpha=alpha,
                alpha_c=alpha_c,
                alpha_s=alpha_s,
                rsd=rsd,
                Lmin=Lmin,
                Lmax=Lmax
            )

            # Call the original function
            return {self.module_name}.populate_galaxies(halos, subhalos,
                    hod_params)
        end
        """
        )

        # Store reference to the optimized function
        self.fast_populate_galaxies = Main.fast_populate_galaxies

        # Optional: Precompile with dummy data for even better performance
        if precompile:
            self._precompile()

    def _precompile(self):
        """
        Precompile the function with dummy data to improve performance for the
        first call.
        """
        try:
            # Create small arrays for precompilation
            dummy_size = 10
            Main.eval(
                f"""
            # Create dummy data for precompilation
            dummy_h_mass = ones(Float32, {dummy_size})
            dummy_h_x = ones(Float32, {dummy_size})
            dummy_h_y = ones(Float32, {dummy_size})
            dummy_h_z = ones(Float32, {dummy_size})
            dummy_h_velocity = ones(Float32, {dummy_size})
            dummy_h_sigma = ones(Float32, {dummy_size})

            dummy_s_mass = ones(Float32, {dummy_size})
            dummy_s_host_velocity = ones(Float32, {dummy_size})
            dummy_s_n_particles = ones(Int32, {dummy_size})
            dummy_s_x = ones(Float32, {dummy_size})
            dummy_s_y = ones(Float32, {dummy_size})
            dummy_s_z = ones(Float32, {dummy_size})
            dummy_s_velocity = ones(Float32, {dummy_size})

            # Precompile with dummy data
            fast_populate_galaxies(
                dummy_h_mass, dummy_h_x, dummy_h_y, dummy_h_z,
                dummy_h_velocity, dummy_h_sigma,
                dummy_s_mass, dummy_s_host_velocity, dummy_s_n_particles,
                dummy_s_x, dummy_s_y, dummy_s_z, dummy_s_velocity,
                12.0, 0.2, 13.0, 1.0, 1.0, 0.3, 0.2, true, 0.0, 100.0
            )
            """
            )
        except Exception as e:
            print(f"Precompilation warning (non-critical): {e}")

    def populate_galaxies(
        self,
        halos: Dict[str, np.ndarray],
        subhalos: Dict[str, np.ndarray],
        hod_params: Dict[str, Union[float, bool]],
    ) -> Dict[str, Union[np.ndarray, int]]:
        """
        Performance-optimized wrapper for the Julia populate_galaxies function.

        Parameters
        ----------
        halos : dict
            Dictionary containing halo properties
        subhalos : dict
            Dictionary containing subhalo properties
        hod_params : dict
            Dictionary containing HOD parameters

        Returns
        -------
        dict
            Dictionary with galaxy positions and total count
        """
        # Pass individual arrays directly to avoid creating temporary
        # NamedTuples in Python
        result = self.fast_populate_galaxies(
            halos["mass"],
            halos["x"],
            halos["y"],
            halos["z"],
            halos["velocity"],
            halos["sigma"],
            subhalos["mass"],
            subhalos["host_velocity"],
            subhalos["n_particles"],
            subhalos["x"],
            subhalos["y"],
            subhalos["z"],
            subhalos["velocity"],
            hod_params["lnMcut"],
            hod_params["sigma"],
            hod_params["lnM1"],
            hod_params["kappa"],
            hod_params["alpha"],
            hod_params["alpha_c"],
            hod_params["alpha_s"],
            hod_params["rsd"],
            hod_params["Lmin"],
            hod_params["Lmax"],
        )

        # Convert result to Python
        return {
            # Assuming the result tuple has x, y, z, count in that order
            "x": np.array(result[0]),
            "y": np.array(result[1]),
            "z": np.array(result[2]),
            "count": int(result[3]),
        }


# Convenience function for easier usage
def populate_galaxies(
    halos_data: Dict[str, np.ndarray],
    subhalos_data: Dict[str, np.ndarray],
    hod_pars: Dict[str, Union[float, bool]],
    julia_path=None,
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Fast convenience function to call the Julia populate_galaxies function.

    Parameters
    ----------
    halos_data : dict
        Dictionary with halo properties
    subhalos_data : dict
        Dictionary with subhalo properties
    hod_pars : dict
        Dictionary with HOD parameters
    julia_path : str, optional
        Path to the Julia project

    Returns
    -------
    dict
        Dictionary with galaxy positions and count
    """
    # Use a global wrapper to avoid initialization costs for repeated calls
    global _wrapper
    if "_wrapper" not in globals():
        _wrapper = GalPopWrapper(julia_path)

    return _wrapper.populate_galaxies(halos_data, subhalos_data, hod_pars)


if __name__ == "__main__":
    # Example data
    halos = {
        "mass": np.array([1e12, 5e12, 2e13], dtype=np.float32),
        "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "y": np.array([15.0, 25.0, 35.0], dtype=np.float32),
        "z": np.array([5.0, 15.0, 25.0], dtype=np.float32),
        "velocity": np.array([100.0, 200.0, 300.0], dtype=np.float32),
        "sigma": np.array([50.0, 70.0, 90.0], dtype=np.float32),
    }

    subhalos = {
        "mass": np.array([1e11, 2e11, 3e11], dtype=np.float32),
        "host_velocity": np.array([100.0, 200.0, 300.0], dtype=np.float32),
        "n_particles": np.array([1000, 2000, 3000], dtype=np.int32),
        "x": np.array([12.0, 22.0, 32.0], dtype=np.float32),
        "y": np.array([17.0, 27.0, 37.0], dtype=np.float32),
        "z": np.array([7.0, 17.0, 27.0], dtype=np.float32),
        "velocity": np.array([120.0, 220.0, 320.0], dtype=np.float32),
    }

    hod_params = {
        "lnMcut": 12.0,
        "sigma": 0.2,
        "lnM1": 13.0,
        "kappa": 1.0,
        "alpha": 1.0,
        "alpha_c": 0.3,
        "alpha_s": 0.2,
        "rsd": True,
        "Lmin": 0.0,
        "Lmax": 100.0,
    }

    # Performance testing
    import time

    start = time.time()
    result = populate_galaxies(halos, subhalos, hod_params)
    end = time.time()

    print(f"Generated {result['count']} galaxies in {end - start:.6f} seconds")
