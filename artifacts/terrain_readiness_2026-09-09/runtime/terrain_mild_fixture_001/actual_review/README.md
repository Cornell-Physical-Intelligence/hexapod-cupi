All 48 mild curriculum fixtures passed the exact Isaac mesh, RayCaster, PhysX query and sphere-contact checks. The recorded host gate was replayed against the unchanged complete source before this review was frozen.

There were 3,888 RayCaster rays, 192 PhysX queries and 144 physical probes; all 48 outside-mesh queries missed. Maximum ray height error was 0.1722 µm, maximum PhysX height error 0.05075 µm and maximum absolute sphere-bottom gap 18.72 µm. Every probe retained contact in all of the final 40 steps. The full log contains no reported incomplete contact or friction data.

The 500 × 5 ms probe integration follows one initial 5 ms query step. Materials were nominal friction 1 and restitution 0. The family split was 18 smooth-rough, 18 ramp, six step and six ridge fixtures, with 32 training and 16 held-out fixtures. These 48 contain no pits.

This proves fixture import and sampled geometry/contact behavior only. No robot, locomotion policy or perception system was qualified. Frozen catalog status remains unchanged; this result is separate admission evidence. The historical audit verified 11 raw payloads, 105 source files, 97 catalog/geometry files, exact owned-container absence and pause045 restoration. It makes no claim about later GPU ownership or locks.

`analyze.py` is the original source-bound replay, using its documented sibling source directory; `report.json` and `review.log` preserve the executed result. The terminal publication's portable verifier checks copied payloads without needing the full source tree.
