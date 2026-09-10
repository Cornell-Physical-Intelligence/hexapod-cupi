# Updated Onshape export: mass, inertia and material audit

Source: `HexapodLegUpdatedV2.zip`, SHA-256 `31f33044285c3de2dbd85180e7c4886c31d80da4270dfd6b94bbd50d150a4cf7`.

**These are unmodified exported CAD mass properties, not a measured physical robot.** The URDF contains one fused link, no joints, no closure records and no named frames. The pickle retains 1,753 part instances across 59 mesh types, sufficient for a mass ledger but not an authoritative joint/rigid-body map.

Total exported CAD mass: **5.147603654203 kg** (URDF writes 5.1476 kg). Gravity is not exported: weight force depends on local gravity; the ledger therefore reports mass.

## Aggregate properties in the export assembly frame

COM [x, y, z], metres: `-0.00145342645877, -0.000136641296123, 0.0366756074749`.

Full inertia about that COM, aligned with export axes, kg·m²:

```text
 0.181685079304  0.000238375507893  0.000945749361462
 0.000238375507893  0.105307454796  0.00147935035808
 0.000945749361462  0.00147935035808  0.279470573085
```

Off-diagonal terms are retained. These values describe the arbitrary exported assembly pose; articulated link inertias require correct rigid-body grouping. Source CAD tensors were rotated by R I Rᵀ and combined using the parallel-axis theorem, never diagonalized or replaced with mesh-derived tensors.

## Mass breakdown

| Category | Part instances | Mass (kg) |
|---|---:|---:|
| catalog fasteners and hardware | 646 | 0.594667318 |
| custom robot structure | 99 | 3.294947516 |
| separate bearing parts | 342 | 0.138473402 |
| vendor motor parts | 666 | 1.119515419 |

Grouping is explicitly based on the source document IDs in `.part` metadata. “Separate bearing parts” are the 18 repeated external bearing assemblies from document `90e95efa79c7cf10f5549d54`, not automatically included in motor replacement mass.

## Motor mass gap

The unique vendor housing mesh appears **18 times**. The 666 vendor part instances total **1.119515418977 kg**, or **62.195301 g per motor**. Every vendor mesh multiplicity is an integer multiple of 18. There are **108 zero-mass motor part instances** across two nonempty mesh types; zero inertia on those parts is algebraically consistent but does not establish correct physical mass.

The repository historical RS05 nominal motor mass is 191 g. **If the exact same 18 motors are present**, replacing the aggregate vendor CAD mass would add 2.318484581 kg and produce a hypothetical robot mass of **7.466088235 kg**. No correction was applied. The motor internals’ distribution, housing/rotor split, cables and hardware need verification before a physically accurate inertia correction can be made. Do not carry forward the old robot total or its old per-link corrections by default. See repository `docs/RS05_SPEC_REVIEW.md` and `docs/CAD_ENGINEER_HANDOFF.md` for the historical evidence.

## Materials: what was and was not exported

**Assigned engineering material names are absent.** The `.part` JSONs provide part identity, source document/microversion, configuration and names. The pickle provides mass, COM, full inertia, mesh geometry references and RGBA appearance. All URDF `<material>` records are visual names plus color; none supply a density, alloy/polymer grade, elastic modulus, Poisson ratio, friction, restitution, or contact stiffness.

Several catalog names mention steel/stainless grades, but those are name hints rather than proof of assigned Onshape materials. The table reports **estimated density = exported CAD mass / tessellated mesh volume** only where the mesh is a valid oriented closed solid. Density, color and part names never establish material identity. Density differences also include tessellation and CAD mass overrides. The user explicitly accepted mass properties without assigned material names; their absence is not an intake blocker. Measured manufactured masses and actual foot/contact properties remain useful for physical validation.

## Per-type ledger

| Mesh type | Count | Each mass (g, min–max) | Total mass (g) | Estimated density (kg/m³, min–max) |
|---|---:|---:|---:|---:|
| `tibia` | 6 | 85.118443–85.118443 | 510.710659 | 1180.643–1180.643 |
| `bottom_enclosure` | 1 | 409.977821–409.977821 | 409.977821 | 2810.155–2810.155 |
| `top_enclosure` | 1 | 404.268920–404.268920 | 404.268920 | 2810.074–2810.074 |
| `1_1_06_eb463_507_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__çð³ý_à_éì1` | 18 | 21.834138–21.834138 | 393.014477 | 2683.628–2683.628 |
| `standoff_plate` | 4 | 95.697716–95.697716 | 382.790863 | 2680.588–2680.588 |
| `motor_bearing_holder` | 18 | 14.515051–14.515051 | 261.270909 | 2703.430–2703.430 |
| `bottom_plate` | 1 | 250.759274–250.759274 | 250.759274 | 2679.916–2679.916 |
| `1_1_02_eb461_502_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 18 | 10.039341–10.039341 | 180.708134 | 2682.930–2682.930 |
| `first_joint_spacer` | 6 | 29.604288–29.604288 | 177.625726 | 1299.861–1299.861 |
| `femur_first_stage` | 6 | 28.932728–28.932728 | 173.596365 | 2680.496–2680.496 |
| `first_joint_bottom_plate` | 6 | 26.471251–26.471251 | 158.827505 | 2680.544–2680.544 |
| `standoff_plate__2` | 2 | 77.020982–77.020982 | 154.041964 | 2680.515–2680.515 |
| `0001755650_00mini_000_asm_1_asm_1_asm_00mini__20250103_a0001_asm_1_as_asm_1_1_30_gr_2202_asm_1_asm_1_1_30_gr_2202_asm_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 18 | 8.098780–8.098780 | 145.778046 | 2683.358–2683.358 |
| `tibia_attatchment_plate` | 6 | 19.443942–19.443942 | 116.663651 | 2680.755–2680.755 |
| `1_1_06_eb463_509_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 18 | 6.371666–6.371666 | 114.689982 | 2681.805–2681.805 |
| `motor_flange` | 18 | 6.284657–6.284657 | 113.123833 | 2683.404–2683.404 |
| `m4_low_profile_socket_head_screw__8mm_l__18_8_ss` | 72 | 1.436027–1.436027 | 103.393970 | 8038.740–8038.740 |
| `m3_low_profile_socket_head_screw__8mm_l__18_8_ss` | 144 | 0.712139–0.712139 | 102.547949 | 8040.710–8040.710 |
| `1_2_16_001425_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__çð³ý_à_éì2` | 18 | 5.650791–5.650791 | 101.714245 | 2682.126–2682.126 |
| `bearing_insert` | 18 | 5.562323–5.562323 | 100.121822 | 2703.430–2703.430 |
| `first_joint_top` | 6 | 13.528034–13.528034 | 81.168202 | 2680.479–2680.479 |
| `m4_countersunk_hex_drive_screw__12mm_l__alloy_steel` | 60 | 1.260632–1.260632 | 75.637903 | 7857.499–7857.499 |
| `1_1_31_eb463_510_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 18 | 3.575505–3.575505 | 64.359095 | 2682.024–2682.024 |
| `6706__300001_1_1_00mini_000_asm_1_asm_1_asm_00mini__20250103_a0001_asm_1_as_asm_1_1_30_gr_2202_asm_1_asm_1_1_30_gr_2202_asm_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 18 | 3.362836–3.362836 | 60.531045 | 2683.484–2683.484 |
| `m3_low_profile__socket_head_shoulder_screw__45mm_shoulder_l__4mm_l__18_8_ss` | 12 | 4.999075–4.999075 | 59.988897 | 7861.066–7861.066 |
| `m3_low_profile_socket_head_screw__6mm_l__18_8_ss` | 96 | 0.621105–0.621105 | 59.626034 | 8039.140–8039.140 |
| `hex_socket_head_countersunk_screw_grade_c_m3x0_5_x_10__1cb4a3dadc02821627dad8d7e6578bbe` | 72 | 0.665400–0.665400 | 47.908835 | 7750.265–7750.265 |
| `revolve1_1` | 18 | 2.580894–2.580894 | 46.456094 | 7856.942–7856.942 |
| `m3_low_profile_socket_head_screw__5mm_l__hs_steel` | 73 | 0.566165–0.566165 | 41.330010 | 7859.732–7859.732 |
| `m4_through_hole_standoff__8mm_l__hardened_steel` | 21 | 1.956843–1.956843 | 41.093698 | 7858.890–7858.890 |
| `pem_m4_self_clinching_nut__300_ss` | 48 | 0.835536–0.835536 | 40.105747 | 7859.976–7859.976 |
| `revolve1_2` | 18 | 1.787560–1.787560 | 32.176086 | 7856.669–7856.669 |
| `m3_countersunk_hex_drive_screw__8mm_l__alloy_steel` | 36 | 0.482892–0.482892 | 17.384095 | 7858.767–7858.767 |
| `0001755652_1_1_30_gr_2202_asm_1_asm_1_1_30_gr_2202_asm_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 36 | 0.470230–0.470230 | 16.928281 | 2683.420–2683.420 |
| `revolve3_1` | 18 | 0.762728–0.762728 | 13.729100 | 7859.973–7859.973 |
| `revolve3_2` | 18 | 0.762728–0.762728 | 13.729100 | 7859.973–7859.973 |
| `1_21_121000022__610001_1_1_00mini_000_asm_1_asm_1_asm_00mini__20250103_a0001_asm_1_as_asm_1_1_30_gr_2202_asm_1_asm_1_1_30_gr_2202_asm_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 18 | 0.646775–0.646775 | 11.641947 | 2684.291–2684.291 |
| `1_2_05_000608_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 72 | 0.125176–0.125176 | 9.012674 | 2680.504–2680.504 |
| `0001755652_00mini_000_asm_1_asm_1_asm_00mini__20250103_a0001_asm_1_as_asm_1_1_30_gr_2202_asm_1_asm_1_1_30_gr_2202_asm_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 18 | 0.470230–0.470230 | 8.464140 | 2683.420–2683.420 |
| `1_2_05_000555_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 144 | 0.050556–0.050556 | 7.280135 | 2682.486–2682.486 |
| `pem_bsos_m3_6____c8b4713a2b299da432156142d6d3e6d8` | 12 | 0.470848–0.470848 | 5.650182 | 7748.476–7748.476 |
| `1_2_05_000438_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__êäèë1` | 144 | 0.037453–0.037453 | 5.393218 | 2681.563–2681.563 |
| `cirpattern1_1` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_10` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_11` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_12` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_2` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_3` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_5` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_9` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `revolve2` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_4` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_6` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_7` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `cirpattern1_8` | 18 | 0.128469–0.128469 | 2.312451 | 7874.956–7874.956 |
| `mirror1` | 18 | 0.064477–0.064477 | 1.160583 | 7872.854–7872.854 |
| `cirpattern2` | 18 | 0.064477–0.064477 | 1.160583 | 7872.856–7872.856 |
| `1_2_05_020051_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__none` | 54 | 0.000000–0.000000 | 0.000000 | 0.000–0.000 |
| `1_2_05_020051_fl46blw10_48v_5n_m_de_2__fl46blw10_48v_5n_m_de__none__2` | 54 | 0.000000–0.000000 | 0.000000 | unavailable |

## Validation

- Source CAD totals match every written fused URDF mass, COM and inertia component within its decimal precision: **True**.
- Part ordering, mesh identity, unit mesh scales and all 1,753 visual/collision transforms verified. Largest translation-component difference: 4.99877354e-07 m; rotation difference: 7.61039475e-06 rad, consistent with rounded URDF pose text.
- All 1753 part tensors are finite, symmetric, positive semidefinite and satisfy the principal-moment triangle inequality within recorded numerical tolerance: **True**. Positive-mass/positive-definite count: 1645/1645.
- Mesh volume estimates unsuitable for density: 1 of 59 types.
- Restricted pickle loader allows only four inert data classes and three NumPy constructors. No arbitrary module loading or source mutation.

## Files and reproduction

- `part_instances.json`: all 1,753 instances with source/assembly transforms, mass, COM, full tensors, tensor checks and appearance.
- `part_instances.csv`: flat per-instance ledger; tensor columns are about each part COM in its own mesh frame.
- `part_types.json` / `part_types.csv`: 59 type groups, exact metadata, summed masses, source tensors and mesh diagnostics.
- `mass_audit.json`: aggregate properties, all rounded-URDF comparisons and explicit uncertainty flags.
- `source_file_hashes.json`: hashes of every input file.

```sh
python audit_onshape_mass_properties.py --source /path/to/extracted-export --out /path/to/audit --archive /path/to/HexapodLegUpdatedV2.zip
```

The optional archive check pins the supplied ZIP; extracted files must retain their bytes and decode filename mojibake as recorded by the intake. No Isaac/GPU run or hardware calibration is claimed by this audit.

## Required next evidence

1. Obtain or independently validate the current mechanism’s mate graph, rigid-link grouping, joint frames and mechanically justified travel bounds; add closure frames only if the current mechanism requires them.
2. Verify the 18 motor model/revisions and actual mass; measure or document per-motor COM, housing/rotor inertia and extra cables/connectors.
3. Retain the exported masses and tensors as requested; compare key manufactured part masses and complete missing/zero-mass motor internals. Assigned material names are optional per the user clarification.
4. Confirm electronics, battery, payload, wiring, feet and fastener completeness by measured mass/COM inventory.
5. Sum these unchanged source properties into each independently validated rigid link, then verify complete URDF/USD mass tensors, constraints, self-collision exclusions and bounded physics probes.
