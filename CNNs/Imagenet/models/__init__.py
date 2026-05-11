# models/__init__.py
import sys

from .ggd_resnet import build_model as build_ggd_resnet
from .ggd_vgg import build_model as build_ggd_vgg
from .GGM_resnet import build_model as build_GGM_resnet

_BUILDERS = {
    "ggd_resnet": build_ggd_resnet,
    "ggd_vgg": build_ggd_vgg,
    "GGM_resnet": build_GGM_resnet,
}


def build_model(name: str, **kwargs):
    name = name.lower().strip()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_BUILDERS.keys())}")

    # 1) Extract the full_precision flag
    full_precision = kwargs.pop("full_precision", False)

    builder_func = _BUILDERS[name]

    # 2) If full precision is requested, bypass customization (monkeypatch customize_model)
    if full_precision:
        module = sys.modules[builder_func.__module__]
        original_customize = getattr(module, "customize_model", None)

        if original_customize is not None:
            module.customize_model = lambda *args, **kwargs: None  # type: ignore[assignment]

        try:
            return builder_func(**kwargs)
        finally:
            if original_customize is not None:
                module.customize_model = original_customize

    # 3) Normal GGD path
    return builder_func(**kwargs)
