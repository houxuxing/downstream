"""模型定义模块 (向后兼容重定向)

此文件已重构，模型现在位于 model/ 包中。
为保持向后兼容性，从 model 包重新导出所有内容。

新代码请直接使用:
    from model import AnatCLClassifier, setup_parameter_freezing
"""

# 向后兼容：从 model 包重新导出
from model.classifier import AnatCLClassifier
from model.freezing import (
    set_frozen_batchnorm_eval,
    setup_parameter_freezing,
)

__all__ = [
    "AnatCLClassifier",
    "set_frozen_batchnorm_eval",
    "setup_parameter_freezing",
]
