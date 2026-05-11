import sys

from .ggd_resnet import build_model as build_ggd_resnet




_BUILDERS = {
    "ggd_resnet": build_ggd_resnet,
    
}



def build_model(name: str, **kwargs):
    name = name.lower().strip()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_BUILDERS.keys())}")

    full_precision = kwargs.pop("full_precision", False)

    builder_func = _BUILDERS[name]

    if full_precision:
        module = sys.modules[builder_func.__module__]
        original_customize = getattr(module, "customize_model", None)

        if original_customize is not None:
            module.customize_model = lambda *args, **kwargs: None

        try:
            return builder_func(**kwargs)
        finally:
            if original_customize is not None:
                module.customize_model = original_customize

    return builder_func(**kwargs)
