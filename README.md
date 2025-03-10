[![Python package](https://github.com/KSU-Cosmo/GalPop/actions/workflows/python-package.yml/badge.svg)](https://github.com/KSU-Cosmo/GalPop/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/KSU-Cosmo/GalPop/graph/badge.svg?token=43UK7SMTWP)](https://codecov.io/gh/KSU-Cosmo/GalPop)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bce28a3599604d0fab140839d8965a45)](https://app.codacy.com/gh/KSU-Cosmo/GalPop/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# GalPop

Code base for population Nbody simulations with galaxies

The main data structure is the following dictionary:

    result = {
        'halo': {
            'mass': [], 'x': [], 'y': [], 'z': [], 'sigma': [], 'velocity': []
        },
        'subsample': {
            'mass': [], 'host_velocity': [], 'n_particles': [],
            'x': [], 'y': [], 'z': [], 'velocity': []
        }
    }

This dictionary contains needed information about halos and satellites.

I mostly work with AbacusSummit simulations, so we provide a code for reading abacus files and converting them into a file that contains this dictionary.
If you are working with other simulations, you have to figure out how to convert the information in those simulations into this format (or get in touch with me, I may be able to help).

* `process_Abacus.py` - contains functions to process AbacusSummit files into this format.
* `process_Abacus.ipynb` - shows how to use those functions.
* `populate_galaxies.jl` - populates the halos with galaxies. I had to write this in Julia, the pure python implementation (populate_galaxies.py) was a bit slow.
* `populate_galaxies_julia.ipynb` - shows how to run this in Julia.
* `populate_galaxies_wrapper.py` - wrapps the call to the Julia function from within python.
* `populate_galaxies.ipynb` - demonstrates how the above works.