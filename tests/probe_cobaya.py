#!/usr/bin/env python3
"""Probe Cobaya internals to find the raw CLASS object."""
import sys
import numpy as np
sys.path.insert(0, '/home/lustre_p/ahmed.omar/workspace/exo_de_project')

from cobaya.model import get_model

info = {
    'params': {
        'a_samp': -40.0,
        's': -160.0,
        'a_exo': {'value': 'lambda a_samp: a_samp'},
        'b_exo': {'value': 'lambda a_samp, s: s - a_samp'},
        'H0': 67.36,
        'omega_b': 0.02237,
        'omega_cdm': 0.1200,
        'n_s': 0.9649,
        'logA': {'value': 3.044, 'drop': True},
        'A_s': {'value': 'lambda logA: 1e-10*np.exp(logA)'},
        'tau_reio': 0.0544,
    },
    'theory': {
        'classy': {
            'extra_args': {
                'z_c_exo': 16.0,
                'sigma_z_exo': 3.25,
                'output': 'mPk',
                'P_k_max_1/Mpc': 360.0,
                'z_max_pk': 20.0,
                'non linear': 'none',
            },
            'ignore_obsolete': True,
        }
    },
    'likelihood': {'one': None},
}

model = get_model(info)

# Force a computation at the reference point
point = {'a_samp': -40.0, 's': -160.0}
model.logpost(point)

# ── Now probe the internals ──
provider = model.provider

print("=== Provider attributes ===")
print([a for a in dir(provider) if not a.startswith('_')])

print("\n=== Does provider.model exist? ===")
print(hasattr(provider, 'model'))

if hasattr(provider, 'model'):
    print("\n=== Model.theory keys ===")
    print(list(provider.model.theory.keys()))

    for name, theory in provider.model.theory.items():
        print(f"\n=== Theory '{name}' ===")
        print(f"  type: {type(theory)}")
        print(f"  has 'classy' attr: {hasattr(theory, 'classy')}")

        if hasattr(theory, 'classy'):
            raw = theory.classy
            print(f"  type of .classy: {type(raw)}")
            print(f"  h()      = {raw.h():.6f}")
            print(f"  Omega_m()= {raw.Omega_m():.6f}")

            # Test P(k,z) directly
            k_test = np.array([0.1])
            z_test = np.array([9.0])
            pk = raw.get_pk_array(k_test, z_test, 1, 1, False)
            print(f"  P(k=0.1, z=9) = {pk[0]:.6e}")

            # Test H(z) and d_A(z) directly
            print(f"  H(z=9)   = {raw.Hubble(9.0):.6e}")
            print(f"  d_A(z=9) = {raw.angular_distance(9.0):.6e}")

            print("\n  ✅ Raw CLASS object accessible and functional")
else:
    print("\n  ❌ provider.model does not exist — need different path")
    print("  Trying alternative: provider._model ...")
    if hasattr(provider, '_model'):
        print("  Found provider._model")
        print(list(provider._model.theory.keys()))
    else:
        print("  ❌ Also not found. Print all provider attributes:")
        for attr in dir(provider):
            if not attr.startswith('__'):
                print(f"    {attr}: {type(getattr(provider, attr, '?'))}")