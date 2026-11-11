"""Pre-paper names (Engine, Platform) must still resolve, with a DeprecationWarning."""
import pytest

import mist
from mist import ClientType, ClusterConfig, LLMClient, MISTClient, MISTCoordinator


def test_old_names_resolve_to_new_objects():
    with pytest.warns(DeprecationWarning):
        from mist import LLMEngine
    assert LLMEngine is LLMClient

    with pytest.warns(DeprecationWarning):
        from mist.Engine import EngineType
    assert EngineType is ClientType

    with pytest.warns(DeprecationWarning):
        from mist.Platforms.platforms import PlatformConfig
    assert PlatformConfig is ClusterConfig

    with pytest.raises(AttributeError):
        mist.NoSuchEngine


def test_old_kwargs_attributes_and_methods():
    with pytest.warns(DeprecationWarning):
        host = MISTClient(model=None, engine_types=[ClientType.HOST], engine_id=7)
    assert host.client_types == [ClientType.HOST] and host.client_id == 7

    with pytest.warns(DeprecationWarning):
        assert host.engine_types == [ClientType.HOST]

    coordinator = MISTCoordinator([], logging_file=None)
    with pytest.warns(DeprecationWarning):
        assert coordinator.add_engine == coordinator.add_client
