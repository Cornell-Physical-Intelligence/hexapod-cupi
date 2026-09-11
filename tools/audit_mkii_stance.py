"""Compatibility for the frozen four-bar validator; implementation: tools.assets.audit_mkii_stance."""
import importlib
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    runpy.run_module('tools.assets.audit_mkii_stance', run_name='__main__')
else:
    _implementation = importlib.import_module('tools.assets.audit_mkii_stance')

    def __getattr__(name):
        return getattr(_implementation, name)

    def __dir__():
        return dir(_implementation)
