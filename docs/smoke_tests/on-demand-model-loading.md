# On-Demand Model Loading

Tests loading model only when needed (reduces idle memory):

```bash
# 1. Enable on_demand_loading for the active engine in config.toml:
#    [whisper]
#    on_demand_loading = true

# 2. Optionally keep the model for N idle seconds after each dictation.
#    idle_unload_secs is a root key and applies to the active engine:
#    idle_unload_secs = 300
#    0 (default) unloads right after every dictation. Retention requires
#    on-demand loading on; Whisper also requires gpu_isolation = false and
#    max_loaded_models > 0.

# 3. Restart daemon
systemctl --user restart voxtype

# 4. Check memory before recording (model not loaded):
systemctl --user status voxtype | grep Memory

# 5. Record and transcribe
voxtype record start && sleep 3 && voxtype record stop

# 6. Check logs for model load/unload:
journalctl --user -u voxtype --since "1 minute ago" | grep -E "Loading|Unloading"
```
