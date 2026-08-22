"""可直接在 PyCharm 里右键运行的实验脚本。

本文件让 ``import experiments.run_xxx`` 也能正常工作：把 ``experiments/`` 自身
加进 ``sys.path``，这样各脚本顶部的 ``import _bootstrap`` 在两种场景下都成立
——既包括 PyCharm 直接运行某个脚本，也包括作为包被导入。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
