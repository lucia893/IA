import subprocess
import sys

# Ejecuta: poetry run python -m src.sweep_run <args_que_manda_wandb>
cmd = ["poetry", "run", "python", "-m", "src.sweep_run"] + sys.argv[1:]
subprocess.run(cmd, check=True)
