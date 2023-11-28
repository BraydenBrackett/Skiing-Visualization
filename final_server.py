import sys
import subprocess
import os

# You will need to use ctrl+c to close the program

data = os.path.normpath(sys.argv[1])

command = ["bokeh", "serve", "--show", "./final.py", "--args", data]
try:
    subprocess.run(command, check=True)
except KeyboardInterrupt:
    sys.exit(0)