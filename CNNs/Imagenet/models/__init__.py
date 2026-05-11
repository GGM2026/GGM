# models/__init__.py
import sys

from .ggm_resnet import build_model as build_ggm_resnet
from .ggm_vgg import build_model as build_ggm_vgg

_BUILDERS = {
    "ggm_resnet": build_ggm_resnet,
    "ggm_vgg": build_ggm_vgg,
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

    # 3) Normal GGM path
    return builder_func(**kwargs)
