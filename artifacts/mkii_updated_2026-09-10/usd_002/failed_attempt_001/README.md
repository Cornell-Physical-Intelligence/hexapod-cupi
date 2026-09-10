# Preserved initial successor attempt

Both USD bundles failed the original CPU audit only at lm_coxa_yaw after the yaw zero moved to -0.515 degrees. Stored quatf expanded to double has norm1.0000000258181416. Gf.Rotation without double normalization yielded a false1.149609e-5 rotation matrix discrepancy; double normalization reduces the discrepancy to2.00633e-10. No source, authored geometry or numeric threshold is changed to resolve this. These initial failed reports remain unchanged.
