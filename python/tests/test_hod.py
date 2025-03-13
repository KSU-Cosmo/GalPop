import os
import sys
import numpy as np
import pytest
from julia import Julia
from julia import Main

# Add the parent directory to the path if needed
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from galpop.hod import populate_galaxies, GalPopWrapper


class TestHODWrapper:
    """Test cases for the HOD Python wrapper."""

    @pytest.fixture(scope="class")
    def julia_setup(self):
        """Set up Julia environment."""
        # Initialize Julia
        jl = Julia(compiled_modules=False)

        # Find the Julia project path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        julia_project_path = os.path.abspath(os.path.join(current_dir, "..", "..", "julia"))

        # Activate the project and load the module
        Main.eval(f'using Pkg; Pkg.activate("{julia_project_path}")')
        Main.eval(f'include(joinpath("{julia_project_path}", "src", "hod.jl"))')

        # Try to determine module structure
        try:
            Main.eval("using .HOD")
            module_name = "HOD"
        except Exception:
            module_name = "Main"

        return {"julia_project_path": julia_project_path, "module_name": module_name}

    @pytest.fixture(scope="class")
    def test_data(self):
        """Create small test datasets for halos and subhalos."""
        # Create small test arrays with fixed random seed for reproducibility
        np.random.seed(42)
        n_halos = 100
        n_subhalos = 200

        # Halo data
        halos = {
            "mass": np.random.uniform(1e12, 1e14, n_halos).astype(np.float32),
            "x": np.random.uniform(0, 100, n_halos).astype(np.float32),
            "y": np.random.uniform(0, 100, n_halos).astype(np.float32),
            "z": np.random.uniform(0, 100, n_halos).astype(np.float32),
            "velocity": np.random.uniform(-500, 500, n_halos).astype(np.float32),
            "sigma": np.random.uniform(50, 200, n_halos).astype(np.float32),
        }

        # Subhalo data
        subhalos = {
            "mass": np.random.uniform(1e10, 1e12, n_subhalos).astype(np.float32),
            "host_velocity": np.random.uniform(-500, 500, n_subhalos).astype(np.float32),
            "n_particles": np.random.randint(100, 1000, n_subhalos).astype(np.int32),
            "x": np.random.uniform(0, 100, n_subhalos).astype(np.float32),
            "y": np.random.uniform(0, 100, n_subhalos).astype(np.float32),
            "z": np.random.uniform(0, 100, n_subhalos).astype(np.float32),
            "velocity": np.random.uniform(-500, 500, n_subhalos).astype(np.float32),
        }

        # HOD parameters
        # Make sure all galaxies make it
        hod_params = {
            "lnMcut": 8.0,
            "sigma": 0.2,
            "lnM1": 11.0,
            "kappa": 1.0,
            "alpha": 1.0,
            "alpha_c": 0.0,
            "alpha_s": 1.0,
            "rsd": False,
            "Lmin": 0.0,
            "Lmax": 100.0,
        }

        return {"halos": halos, "subhalos": subhalos, "hod_params": hod_params}

    def test_wrapper_output_matches_julia(self, julia_setup, test_data):
        """Test that the Python wrapper output matches the pure Julia output."""
        # Extract data from fixtures
        julia_project_path = julia_setup["julia_project_path"]
        module_name = julia_setup["module_name"]
        halos = test_data["halos"]
        subhalos = test_data["subhalos"]
        hod_params = test_data["hod_params"]

        # Run the Python wrapper
        py_result = populate_galaxies(halos, subhalos, hod_params, julia_project_path)

        # Create Julia NamedTuples for the test data
        Main.eval(
            """
        function create_test_tuples(h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
                                    s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
                                    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s, rsd, Lmin, Lmax)
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
            
            return halos, subhalos, hod_params
        end
        """
        )

        # Run the pure Julia implementation
        Main.h_mass = Main.Array(halos["mass"])
        Main.h_x = Main.Array(halos["x"])
        Main.h_y = Main.Array(halos["y"])
        Main.h_z = Main.Array(halos["z"])
        Main.h_velocity = Main.Array(halos["velocity"])
        Main.h_sigma = Main.Array(halos["sigma"])

        Main.s_mass = Main.Array(subhalos["mass"])
        Main.s_host_velocity = Main.Array(subhalos["host_velocity"])
        Main.s_n_particles = Main.Array(subhalos["n_particles"])
        Main.s_x = Main.Array(subhalos["x"])
        Main.s_y = Main.Array(subhalos["y"])
        Main.s_z = Main.Array(subhalos["z"])
        Main.s_velocity = Main.Array(subhalos["velocity"])

        Main.eval(
            f"""
        halos, subhalos, hod_params = create_test_tuples(
            h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
            s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
            {hod_params["lnMcut"]}, {hod_params["sigma"]}, {hod_params["lnM1"]},
            {hod_params["kappa"]}, {hod_params["alpha"]}, {hod_params["alpha_c"]},
            {hod_params["alpha_s"]}, {str(hod_params["rsd"]).lower()}, 
            {hod_params["Lmin"]}, {hod_params["Lmax"]}
        )
        
        julia_result = {module_name}.populate_galaxies(halos, subhalos, hod_params)
        """
        )

        # Extract Julia results
        julia_x = np.array(Main.eval("julia_result.x"))
        julia_y = np.array(Main.eval("julia_result.y"))
        julia_z = np.array(Main.eval("julia_result.z"))
        julia_count = int(Main.eval("julia_result.count"))

        # Compare results
        assert py_result["count"] == julia_count, "Galaxy counts don't match"

        # Since there could be randomness in the HOD, sort arrays before comparing
        # to account for potential different ordering
        print(np.sort(py_result["x"]))
        print(np.sort(julia_x))
        assert np.allclose(np.sort(py_result["x"]), np.sort(julia_x)), "X coordinates don't match"
        assert np.allclose(np.sort(py_result["y"]), np.sort(julia_y)), "Y coordinates don't match"
        assert np.allclose(np.sort(py_result["z"]), np.sort(julia_z)), "Z coordinates don't match"

        print(
            f"Test passed! Both implementations produced {py_result['count']} galaxies with matching coordinates."
        )

    def test_performance(self, julia_setup, test_data):
        """Test the performance of the Python wrapper."""
        import time

        # Extract data from fixtures
        julia_project_path = julia_setup["julia_project_path"]
        halos = test_data["halos"]
        subhalos = test_data["subhalos"]
        hod_params = test_data["hod_params"]

        # Warm-up run
        _ = populate_galaxies(halos, subhalos, hod_params, julia_project_path)

        # Timed run
        start_time = time.time()
        result = populate_galaxies(halos, subhalos, hod_params, julia_project_path)
        elapsed = time.time() - start_time

        print(f"Performance test: Generated {result['count']} galaxies in {elapsed:.6f} seconds")

        # No assertion - this is just for performance tracking
        assert True
