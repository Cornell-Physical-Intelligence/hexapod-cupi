"""Standard-library-only identity for the unchanged RR preload experiment."""
PROPOSAL = {
    'name': 'rr_first_confirmed_landing_preload_001',
    'leg': 'rr', 'leg_index': 5, 'offset_z_m': -.0005, 'duration_s': .30,
    'trigger': 'first RR measured landing completion, before target-time advance',
    'scope': 'one left-strafe case only; not an RM or arc correction',
    'source_of_amplitude': 'directional002 measured target-minus-toe Z change about0.492mm',
    'not_physics_admitted': True, 'old_actor_or_observation_binding_permitted': False,
}
