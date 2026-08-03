"""Punto de entrada para los despliegues que ejecutan desde la raíz."""

import os
import runpy
import sys


BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ELSEC")
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)
runpy.run_path(os.path.join(BACKEND_DIR, "server_ws.py"), run_name="__main__")
