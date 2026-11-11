"""Deprecated pre-paper names, kept as aliases for one release.

MIST now uses the SC'26 paper's terms: an inference *Client* (was "Engine")
runs on a hardware *Cluster* (was "Platform"); the coordinator's events are
*Stage-Push* and *Client-Step*. Old module paths, class/function names,
keyword arguments and attributes still work but emit DeprecationWarning.
"""
import functools
import importlib
import sys
import warnings

# Old module path -> new module path.
MODULES = {
    "mist.Engine": "mist.Client",
    "mist.Engine.MIST_Engine": "mist.Client.MIST_Client",
    "mist.Engine.Mixed_LLM_Engine": "mist.Client.Mixed_LLM_Client",
    "mist.Engine.KV_Retrieval_Engine": "mist.Client.KV_Retrieval_Client",
    "mist.Engine.RAG_Engine": "mist.Client.RAG_Client",
    "mist.Platforms": "mist.Clusters",
    "mist.Platforms.platforms": "mist.Clusters.clusters",
    "mist.Platforms.vllm_platform": "mist.Clusters.vllm_cluster",
    "mist.Platforms.trace_platforms": "mist.Clusters.trace_clusters",
    "mist.Platforms.memory_platform": "mist.Clusters.memory_cluster",
    "mist.Platforms.performance_model": "mist.Clusters.performance_model",
    "mist.Platforms.linear_prediction": "mist.Clusters.linear_prediction",
}

# Keyword arguments whose new name is not the plain engine->client /
# platform->cluster substitution.
KWARGS = {"engine_type": "cluster_type"}


def _new_name(old: str) -> str:
    return (old.replace("Engine", "Client").replace("engine", "client")
               .replace("ENGINE", "CLIENT").replace("Platform", "Cluster")
               .replace("platform", "cluster"))


def _is_old(name: str) -> bool:
    return any(w in name for w in ("Engine", "engine", "ENGINE", "Platform", "platform"))


def _warn(old: str, new: str) -> None:
    warnings.warn(f"'{old}' is deprecated; use '{new}'", DeprecationWarning, stacklevel=3)


def _module_getattr(module):
    def __getattr__(name):
        if _is_old(name):
            new = _new_name(name)
            if new in module.__dict__:
                _warn(name, new)
                return module.__dict__[new]
        raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}")
    return __getattr__


def _instance_getattr(self, name):
    # Only reached when normal lookup fails; never recurse on half-built objects.
    if _is_old(name):
        new = _new_name(name)
        try:
            value = object.__getattribute__(self, new)
        except AttributeError:
            pass
        else:
            _warn(name, new)
            return value
    raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


def _accept_old_kwargs(fn, overrides):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for old in [k for k in kwargs if _is_old(k)]:
            new = overrides.get(old, _new_name(old))
            _warn(old, new)
            kwargs[new] = kwargs.pop(old)
        return fn(*args, **kwargs)
    return wrapper


def install() -> None:
    from mist.Client import MISTClient, ClientMetrics
    from mist.Clusters import ClusterConfig, MemoryCacheConfig
    from mist.Coordinator import MISTCoordinator
    from mist.Request.request import Request, RequestMetrics, StageMetrics

    for old, new in MODULES.items():
        module = importlib.import_module(new)
        sys.modules.setdefault(old, module)
        module.__getattr__ = _module_getattr(module)
    for name in ("mist", "mist.Coordinator", "mist.Global_Network"):
        sys.modules[name].__getattr__ = _module_getattr(sys.modules[name])

    for cls in (MISTClient, ClientMetrics, ClusterConfig, MemoryCacheConfig,
                MISTCoordinator, Request, RequestMetrics, StageMetrics):
        setattr(cls, "__getattr__", _instance_getattr)

    # Constructors (each subclass defines its own __init__) and add_client.
    for base in (MISTClient, ClusterConfig, MISTCoordinator):
        for cls in [base, *_all_subclasses(base)]:
            for method in ("__init__", "add_client"):
                if method in cls.__dict__:
                    setattr(cls, method, _accept_old_kwargs(cls.__dict__[method], KWARGS))


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)
