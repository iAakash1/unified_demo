# Mango models

The only mango model is the knowledge-grounded Qwen2.5-VL MLX QLoRA of project 09. It shares its adapter code with
the tomato MLX model: `models/common/mlx_plantdx.py`, called with `{"crop": "mango", "run": "qwen25vl_mango_qlora"}`
(registry entry `mango_mlx_kg`). It runs only on macOS with Apple Silicon.
