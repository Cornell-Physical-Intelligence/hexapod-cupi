"""Derive a standing-only diagnostic entrypoint from immutable009 rollout mechanics."""
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'reference_physics_adapter_009/source_009/tools/run_reference_physics.py').read_text()
s=s.replace('Bounded full-C zero-residual standing and measured-contact wave proof only.','Matched-origin standing acquisition; quiet/rate outcomes stay separate from admission.')
s=s.replace('from screen_contract import OPTIONS, WAVE, STARTUP, preflight, save, digest','from screen_contract import OPTIONS, STARTUP, save, digest\nfrom origin_contract import CASES, PROTOCOL, preflight, load_initial')
s=s.replace("choices=('standing', 'wave')","choices=('origin',)")
s=s.replace("parser.add_argument('--admission', type=Path)","parser.add_argument('--admission', type=Path, required=True)\nparser.add_argument('--case', choices=tuple(CASES), required=True)")
s=s.replace('from screen_metrics import standing_screen, standing_quiet_review, physical_metrics, measured_flight_touchdowns, measured_progress','from screen_metrics import standing_screen\nfrom omni_quiet_review import quiet_metrics, QUIET_GATES\nfrom matched_origin import patched_factory, initial_readback, ground_readback')
s=s.replace("scope='fullC zero-residual bounded contact-physics screen'","scope='matched-origin diagnostic acquisition; never wave/PPO admission', origin_protocol=PROTOCOL, case=args.case")
s=s.replace('        env, manifest, plan, record, stance, layout, asset_audit, controller_contract = build_reference_environment(args, OPTIONS)',"        payload=load_initial(SOURCE)\n        with patched_factory(payload,args.case):\n            env, manifest, plan, record, stance, layout, asset_audit, controller_contract = build_reference_environment(args, OPTIONS)")
s=s.replace('        env.episode_length_buf.zero_()',"        env.episode_length_buf.zero_()\n        state['initial_readback']=initial_readback(env,payload,args.case)\n        save(args.output/'initial_readback.json',state['initial_readback'])\n        state['ground_readback']=ground_readback(env)\n        save(args.output/'ground_readback.json',state['ground_readback'])\n        if not state['initial_readback']['passed'] or not state['ground_readback']['passed']:\n            raise RuntimeError('Matched reset or infinite-plane readback failed before control0')")
s=s.replace("'canonical_hold_before_scoring_s':STARTUP['canonical_hold_before_scoring_s']}","'canonical_hold_before_scoring_s':STARTUP['canonical_hold_before_scoring_s'],\n            'physical_reset_preserved':False,'reset_scope':'explicit common audited009env6 initial state inside cold reset only'}")
start=s.index("                if args.mode == 'wave' and step >= WAVE['settle_steps']:")
end=s.index('                env.set_evaluation_targets',start)
s=s[:start]+"                startup_ref = startup.sample(step+1)\n                env.set_reference_targets(startup_ref['q_ref'],startup_ref['analytic_velocity_rad_s'],\n                    startup_ref['analytic_acceleration_rad_s2'],startup_ref['valid'])\n"+s[end:]
start=s.index("                if args.mode == 'wave' and step >= WAVE['settle_steps']:")
end=s.index('                if (step+1) % 100',start)
s=s[:start]+s[end:]
start=s.index('        gate = None\n');end=s.index('        passed = bool(',start)
s=s[:start]+"""        gate = None; quiet = None
        if len(samples) > 200:
            gate = standing_screen(data)
            quiet = quiet_metrics(data,0,200,list(layout['joint_names_runtime']),env.step_dt)
            quiet['bounds']=QUIET_GATES
        state['unchanged_physical_gate']=gate
        state['unchanged_quiet_gate']=quiet
        state['all_existing_bounds_met']=bool(gate and gate['passed'] and quiet and quiet['pass'] and quiet['window_duration_s']>=10.)
        state['quiet_failure_does_not_become_an_admission']=True
        state['velocity_fidelity_qualified']=False
"""+s[end:]
# Drop the original unreachable wave gate remainder after the first matched location above.
# end had targeted its first nested passed=bool, so assert the actual final section shape below.
start=s.index('        passed = bool(')
end=s.index('    except Exception as exc:',start)
s=s[:start]+"""        passed = bool(failure is None and len(samples)==args.steps and gate and gate['passed'])
        state.update(status='completed' if passed else 'rejected', failure=failure,
                     acquisition_complete=passed, control_steps=len(samples), finished_unix=time.time())
        save(args.output / 'state.json', state)
        return 0 if passed else 1
"""+s[end:]
s=s.replace('        generator = None\n','').replace('2 if generator is not None else (0 if step+1 < startup.steps else 1)','0 if step+1 < startup.steps else 1')
out=p/'run_origin_physics.py'
if out.exists():raise FileExistsError(out)
out.write_text(s)
