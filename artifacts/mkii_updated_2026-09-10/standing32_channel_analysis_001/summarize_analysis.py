"""Derive compact facts from the preserved result; this does not rerun raw analysis."""
import statistics


def summarize(a):
    r = a['native_report_unchanged']['replicas']
    rates = [(e['env'], name, v) for e in a['rates_postsettle400Hz_float64_accumulation'] for name, v in e['joints'].items()]
    def peak(key):
        e, name, v = max(rates, key=lambda x: abs(x[2][key]))
        return {'env': e, 'joint': name, 'value': v[key]}
    return {
        'schema': 'standing32_channel_public_summary_v1',
        'original_admission_unchanged': {
            'all_pass': a['original_standing_pass'], 'replicas': len(r),
            'combined_pass': sum(x['pass'] for x in r),
            'physical_pass': sum(not x['failed_physical_bounds'] for x in r),
            'quiet_pass': sum(x['quiet']['pass'] for x in r),
            'support_failure_replicas': sum(x['physical']['post_settle_missing_six_toe_substeps'] > 0 for x in r),
            'original_quiet_window_s': sorted(set(x['quiet']['window_duration_s'] for x in r)),
            'original_quiet_samples_50Hz': sorted(set(x['quiet']['window_samples'] for x in r)),
            'original_max_sdk_quiet_rms_rad_s': max(x['quiet']['max_joint_velocity_rms_rad_s'] for x in r),
        },
        'event_channels': a['event_channel_summary_observations_only'],
        'exact_pre_post_angle_scalar_comparisons': a['exact_pre_post_angle_scalar_comparisons'],
        'exact_patch_reconstruction_rows': a['patch_FP32_product_FP64_sum_exact_rows_all32'],
        'diagnostic_rates_same_16s_400Hz': {
            'joint_environment_pairs': len(rates),
            'maximum_sdk_rms_rad_s': peak('sdk_rms'),
            'maximum_interval_angle_rms_rad_s': peak('interval_angle_rms'),
            'maximum_absolute_sdk_minus_interval_integral_rad': peak('sdk_minus_interval_integral_rad'),
            'median_sdk_rms_rad_s': statistics.median(x[2]['sdk_rms'] for x in rates),
            'median_interval_angle_rms_rad_s': statistics.median(x[2]['interval_angle_rms'] for x in rates),
            'absolute_integral_difference_over_0_01_rad_diagnostic_bin_only': sum(abs(x[2]['sdk_minus_interval_integral_rad']) > .01 for x in rates),
        },
        'recorded_slots': {'capacity': a['recorded_capacity'], 'all_rows': a['pressure_summary_all'], 'event_rows': a['pressure_summary_event_rows'], 'global_backend_occupancy_unknown': True},
        'consumed_raw_files': len(a['consumed_raw_inputs']),
        'consumed_raw_bytes': sum(x['size_bytes'] for x in a['consumed_raw_inputs'].values()),
        'raw_and_source_reverified_after_analysis': a['raw_and_source_reverified_after_analysis'],
        'original_raw_included_in_this_bundle': False,
        'no_new_native_execution_or_policy_admission': True,
    }
