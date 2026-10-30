# GPU memory comparison report

_No GPU run has been recorded in this checkout yet._

Cloud Agent VMs are CPU-only. Produce this report on a CUDA host (Jetson Orin
recommended):

```bash
# On Jetson Orin
bash scripts/setup_jetson_orin.sh
uv run python benchmark/scripts/run_gpu_memory_compare.py
```

See [jetson_orin.md](jetson_orin.md) for environment setup, memory-conservative
configs, and self-hosted worker notes.

After a successful run, this file is overwritten with ranked peak VRAM results.
