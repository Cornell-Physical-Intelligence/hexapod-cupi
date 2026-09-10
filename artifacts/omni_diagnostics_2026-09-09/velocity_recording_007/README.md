# Actual rejected-policy recording 007

**The recording completed; the policy remains rejected and Stage 2 remains incomplete.** This is the actual full-C robot under the final 50-update pilot003 actor, with ground command arrows, labels and an actual-travel trail. No reference moves the simulated body. The single continuous episode contains 34 seconds, 1,700 controls and 850 frames at 25 fps. It completes without a terminal event; the short quiet windows do not overturn the separate 96-replica quiet/stop failure.

[Play the recording](run/recording/rollout.mp4). [Six-frame contact sheet](contact_sheet.jpg). [Measured recording review](recording_review.json).

Maximum planar excursion from the first post-step pose is only 5.08 mm despite 0.10 m/s translation commands. This video exposes the failed controller; it is not a walking benchmark or Stage 2 admission. The independent [pilot evaluation](../velocity_pilot_003/README.md) records essentially no useful tracking and 0/48 quiet plus 0/48 stop passes after training.

Checkpoint SHA-256: `88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247`. Video SHA-256: `f056266fbe8439c67f86a6dc2fa1e8c1311f47dbdb5037529f27ca5f7b5a23e3`. Strict model loading, original 909-file source and all 553 admitted pilot inputs were verified before and after recording. Full raw telemetry/configuration is retained in [video.json](run/recording/video.json) and trace.npz. The periodic progress.json is a progress snapshot; video.json and campaign.json record successful completion.

Attempt004 failed on initial blank RGB, while005/006 stalled before native Kit AppReady. This successor uses the previously successful training import order, bounded render-only camera warmup and a 90-second AppReady deadline. First render was blank, the second valid, without physics steps during warmup. Successful startup after this change does not prove the native stall's cause. Adapter17 and host5 CPU tests passed. Those earlier attempts remain separate immutable evidence.

[Remote audit](remote_audit.json) binds all20 raw run/restoration payloads, confirms source/input hashes, exact container absence and pause026 timer restoration. Adapter and host retain their original freeze maps. Root decoded the video and visually inspected its first frame and six sample frames before sharing it. This is serial C-study evidence, not qualification of the separate production four-bar robot.
