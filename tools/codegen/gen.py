"""Compatibility entry point.

The implementation lives in main.py and the per-concern generator modules
(model / api_types / gen_*), mirroring PyTorch's torchgen package layout.
CMake invokes this file directly as a script, so support both usages.
"""

import os
import sys

if __package__ in (None, ''):
    # gen.py lives at <root>/tools/codegen/, so three dirnames up is the repo
    # root that makes `tools.codegen.main` importable.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools.codegen.main import main
else:
    from .main import main

if __name__ == '__main__':
    main()
