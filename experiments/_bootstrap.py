"""让 ``experiments/`` 下的脚本无论从哪个目录启动都能跑。

PyCharm 里右键 Run 一个子目录里的脚本时，有两件事默认不成立：

1. 仓库根目录不在 ``sys.path`` 里，``import configs.config`` 会失败；
2. 工作目录未必是仓库根，``data/instances``、``result/`` 这类相对路径会指错地方。

各脚本第一行 ``import _bootstrap`` 即可把这两件事一次修好，用户不需要在
PyCharm 的运行配置里改任何东西。
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 相对路径（data/instances、result/、configs/exp/...）一律以仓库根为基准。
if os.path.abspath(os.getcwd()) != REPO_ROOT:
    os.chdir(REPO_ROOT)
