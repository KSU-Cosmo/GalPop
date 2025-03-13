import os
import sys
import numpy as np
from julia import Julia
from julia import Main
import unittest

# Add the parent directory to the path so we can import the galpop module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from galpop.hod import populate_galaxies, GalPopWrapper


class TestHODWrapper(unittest.TestCase):
    """Test cases for the HOD Python wrapper."""

    def setUp(self):
        """Set up test fixtures, including initializing Julia environment."""
        # Initialize Julia
        jl = Julia(compiled_modules=False)

        # Find the Julia project path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.julia_project_path = os.path.abspath(
            os.path.join(current_dir, "..", "..", "..", "julia")
        )

        # Activate the project and load the module
        Main.eval(f'using Pkg; Pkg.activate("{self.julia_project_path}")')
        Main.eval(f'include(joinpath("{self.julia_project_path}", "src", "hod.jl"))')

        # Try to determine module structure
        try:
            Main.eval("using .HOD")
            self.module_name = "HOD"
        except Exception:
            self.module_name = "Main"

        # Create test data
        self.create_test_data()

    def create_test_data(self):
        """Create small test datasets for halos and subhalos."""
        # Create small test arrays
        n_halos = 100
        n_subhalos = 200

        # Halo data
        self.halos = {
            "mass": np.random.uniform(1e12, 1e14, n_halos).astype(np.float32),
            "x": np.random.uniform(0, 100, n_halos).astype(np.float32),
            "y": np.random.uniform(0, 100, n_halos).astype(np.float32),
            "z": np.random.uniform(0, 100, n_halos).astype(np.float32),
            "velocity": np.random.uniform(-500, 500, n_halos).astype(np.float32),
            "sigma": np.random.uniform(50, 200, n_halos).astype(np.float32),
        }

        # Subhalo data
        self.subhalos = {
            "mass": np.random.uniform(1e10, 1e12, n_subhalos).astype(np.float32),
            "host_velocity": np.random.uniform(-500, 500, n_subhalos).astype(np.float32),
            "n_particles": np.random.randint(100, 1000, n_subhalos).astype(np.int32),
            "x": np.random.uniform(0, 100, n_subhalos).astype(np.float32),
            "y": np.random.uniform(0, 100, n_subhalos).astype(np.float32),
            "z": np.random.uniform(0, 100, n_subhalos).astype(np.float32),
            "velocity": np.random.uniform(-500, 500, n_subhalos).astype(np.float32),
        }

        # HOD parameters
        self.hod_params = {
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

    def test_wrapper_output_matches_julia(self):
        """Test that the Python wrapper output matches the pure Julia output."""
        # Run the Python wrapper
        py_result = populate_galaxies(
            self.halos, self.subhalos, self.hod_params, self.julia_project_path
        )

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
        Main.h_mass = Main.Array(self.halos["mass"])
        Main.h_x = Main.Array(self.halos["x"])
        Main.h_y = Main.Array(self.halos["y"])
        Main.h_z = Main.Array(self.halos["z"])
        Main.h_velocity = Main.Array(self.halos["velocity"])
        Main.h_sigma = Main.Array(self.halos["sigma"])

        Main.s_mass = Main.Array(self.subhalos["mass"])
        Main.s_host_velocity = Main.Array(self.subhalos["host_velocity"])
        Main.s_n_particles = Main.Array(self.subhalos["n_particles"])
        Main.s_x = Main.Array(self.subhalos["x"])
        Main.s_y = Main.Array(self.subhalos["y"])
        Main.s_z = Main.Array(self.subhalos["z"])
        Main.s_velocity = Main.Array(self.subhalos["velocity"])

        Main.eval(
            f"""
        halos, subhalos, hod_params = create_test_tuples(
            h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
            s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
            {self.hod_params["lnMcut"]}, {self.hod_params["sigma"]}, {self.hod_params["lnM1"]},
            {self.hod_params["kappa"]}, {self.hod_params["alpha"]}, {self.hod_params["alpha_c"]},
            {self.hod_params["alpha_s"]}, {str(self.hod_params["rsd"]).lower()}, 
            {self.hod_params["Lmin"]}, {self.hod_params["Lmax"]}
        )
        
        julia_result = {self.module_name}.populate_galaxies(halos, subhalos, hod_params)
        """
        )

        # Extract Julia results
        julia_x = np.array(Main.eval("julia_result.x"))
        julia_y = np.array(Main.eval("julia_result.y"))
        julia_z = np.array(Main.eval("julia_result.z"))
        julia_count = int(Main.eval("julia_result.count"))

        # Compare results
        self.assertEqual(py_result["count"], julia_count, "Galaxy counts don't match")

        # Since there could be randomness in the HOD, sort arrays before comparing
        # to account for potential different ordering
        self.assertTrue(
            np.allclose(np.sort(py_result["x"]), np.sort(julia_x)), "X coordinates don't match"
        )
        self.assertTrue(
            np.allclose(np.sort(py_result["y"]), np.sort(julia_y)), "Y coordinates don't match"
        )
        self.assertTrue(
            np.allclose(np.sort(py_result["z"]), np.sort(julia_z)), "Z coordinates don't match"
        )

        print(
            f"Test passed! Both implementations produced {py_result['count']} galaxies with matching coordinates."
        )

    def test_performance(self):
        """Test the performance of the Python wrapper."""
        import time

        # Warm-up run
        _ = populate_galaxies(self.halos, self.subhalos, self.hod_params, self.julia_project_path)

        # Timed run
        start_time = time.time()
        result = populate_galaxies(
            self.halos, self.subhalos, self.hod_params, self.julia_project_path
        )
        elapsed = time.time() - start_time

        print(f"Performance test: Generated {result['count']} galaxies in {elapsed:.6f} seconds")


if __name__ == "__main__":
    unittest.main()
