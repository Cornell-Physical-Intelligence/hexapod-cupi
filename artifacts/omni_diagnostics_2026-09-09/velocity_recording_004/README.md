# Actual progress recorder attempt004

**Failed recording attempt; Stage2 remains incomplete.** Strict model load and full-robot construction passed; first RGB frame was blank before any control rollout. No video was produced.

The actual intended actor is the rejected50-update pilot003 checkpoint `88f8e60a7b1536229f51524cdc646f06d53e4f33cabb74aa89ebde73d06f7247`. See [pilot evidence](../velocity_pilot_003/README.md). Source003 and all553admitted pilot inputs remained unchanged. Raw run/pause files match the remote SHA map; both exact container ID and name are absent, and pause023 restored the previously active forecasting timers. This snapshot does not claim the GPU is idle during a later attempt.

[Campaign](run/campaign.json), [complete log](run/logs/recording.log), [independent remote audit](remote_audit.json), [adapter source](adapter/README.md). The adapter and host each retain their original freeze maps. Installed systemPython passed the renderer tests; the locked uv environment lacked imageio, which was recorded as a test dependency limitation without changing project dependencies.

Attempts005/006 remain infrastructure failures, not additional policy evaluations. The underlying pilot is separately rejected for no useful locomotion and degraded quiet standing. Original004 source and all later attempts are separate immutable artifacts.
