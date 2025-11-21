# run_with_poetry.py
import os
import subprocess
import sys

cmd = ["poetry", "run", "python"] + sys.argv[1:]
subprocess.run(cmd)
