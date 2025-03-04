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

* `process_Abacus.py` - is the file that contains functions to process AbacusSummit files into this format.
* `process_Abacus.ipynb` - is the notebook that shows how to use those functions.
