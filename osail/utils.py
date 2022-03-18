import importlib
import typing as th
import torch


def import_context(name: str):
    return importlib.import_module(name)


def get_value(name: str, context: th.Optional[th.Any] = None, strict: bool = True):
    var = context if context is not None else import_context(name.split('.')[0])
    for split in name.split('.')[(0 if context is not None else 1):]:
        if isinstance(var, dict):
            if split not in var:
                if strict:
                    raise KeyError('Invalid key "%s"' % name)
                else:
                    return None
            var = var[split]
        else:
            if not hasattr(var, split):
                if strict:
                    raise AttributeError('Invalid attribute %s' % name)
                else:
                    return None
            var = getattr(var, split)
    return var


def freeze_params(model: torch.nn.Module):
    old_states = []
    for param in model.parameters():
        old_states.append(param.requires_grad)
        param.requires_grad = False
    return old_states


def unfreeze_params(model: torch.nn.Module, old_states: th.Optional[th.List[bool]] = None):
    for idx, param in enumerate(model.parameters()):
        param.requires_grad = True if old_states is None else old_states[idx]
