"""Compatibility for the frozen four-bar validator; implementation: tools.assets.mkii_fourbar_kinematics."""
import importlib
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    runpy.run_module('tools.assets.mkii_fourbar_kinematics', run_name='__main__')
else:
    _implementation = importlib.import_module('tools.assets.mkii_fourbar_kinematics')

    def __getattr__(name):
        return getattr(_implementation, name)

    def __dir__():
        return dir(_implementation)
