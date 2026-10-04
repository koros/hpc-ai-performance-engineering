import ctypes
import sys
import types

import pytest

from hpc_ai_perf.profiling import system


class _DriverFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def _fake_cuda_driver(device_uuids: tuple[bytes, ...], *, reported_count: int | None = None):
    def get_count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = (
            len(device_uuids) if reported_count is None else reported_count
        )
        return 0

    def get_device(pointer, ordinal):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = ordinal
        return 0

    def get_uuid(pointer, device):
        uuid = ctypes.cast(pointer, ctypes.POINTER(system._CudaUuid)).contents
        for index, value in enumerate(device_uuids[device]):
            uuid.bytes[index] = value
        return 0

    return types.SimpleNamespace(
        cuInit=_DriverFunction(lambda flags: 0),
        cuDeviceGetCount=_DriverFunction(get_count),
        cuDeviceGet=_DriverFunction(get_device),
        cuDeviceGetUuid_v2=_DriverFunction(get_uuid),
    )


@pytest.fixture
def torch_without_cuda_uuids(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_properties=lambda index: types.SimpleNamespace(name="NVIDIA L40S"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")


def test_cuda_driver_uuids_select_two_owned_gpus_from_eight_smi_rows(monkeypatch, torch_without_cuda_uuids):
    first = bytes(range(16))
    second = bytes(range(16, 32))
    monkeypatch.setattr(system.ctypes, "CDLL", lambda name: _fake_cuda_driver((first, second)))

    target = system.visible_cuda_device_uuids(2)
    rows = [f"GPU-{index:032x}, 1" for index in range(6)]
    rows.insert(1, f"{target[1]}, 20")
    rows.insert(6, f"{target[0]}, 10")

    selected, method = system.select_nvidia_smi_rows(rows, target_gpu_uuids=target, expected_gpu_count=2)

    assert target == (
        "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f",
        "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f",
    )
    assert selected == [f"{target[0]}, 10", f"{target[1]}, 20"]
    assert method == "cuda_uuid"


def test_cuda_driver_count_mismatch_keeps_eight_row_mapping_unsafe(monkeypatch, torch_without_cuda_uuids):
    monkeypatch.setattr(system.ctypes, "CDLL", lambda name: _fake_cuda_driver((bytes(range(16)),), reported_count=8))

    target = system.visible_cuda_device_uuids(2)

    assert target == ()
    with pytest.raises(ValueError, match="cannot prove nvidia-smi device ownership"):
        system.select_nvidia_smi_rows(
            [f"GPU-{index:032x}, 1" for index in range(8)],
            target_gpu_uuids=target,
            expected_gpu_count=2,
        )


def test_cuda_driver_legacy_uuid_symbol_is_supported(monkeypatch, torch_without_cuda_uuids):
    driver = _fake_cuda_driver((bytes(range(16)), bytes(range(16, 32))))
    driver.cuDeviceGetUuid = driver.cuDeviceGetUuid_v2
    del driver.cuDeviceGetUuid_v2
    monkeypatch.setattr(system.ctypes, "CDLL", lambda name: driver)

    assert len(system.visible_cuda_device_uuids(2)) == 2


def test_cuda_driver_failure_keeps_uuid_mask_fallback(monkeypatch, torch_without_cuda_uuids):
    monkeypatch.setattr(system.ctypes, "CDLL", lambda name: _fake_cuda_driver((bytes(range(16)),), reported_count=8))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one,GPU-two")

    assert system.visible_cuda_device_uuids(2) == ("GPU-one", "GPU-two")
