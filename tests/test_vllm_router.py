import pytest

from hpc_ai_perf.workloads.vllm_router import build_gpu_groups, route_prompt_indices


def test_build_gpu_groups_partitions_four_gpus_into_two_tp2_replicas() -> None:
    assert build_gpu_groups(gpu_count=4, tensor_parallel_size=2, replica_count=2) == (
        ("0", "1"),
        ("2", "3"),
    )


def test_build_gpu_groups_preserves_scheduler_visible_device_ids() -> None:
    assert build_gpu_groups(
        gpu_count=4,
        tensor_parallel_size=2,
        replica_count=2,
        visible_devices="4,6,7,9",
    ) == (("4", "6"), ("7", "9"))


@pytest.mark.parametrize("routing", ["round_robin", "least_loaded"])
def test_route_prompt_indices_balances_requests_across_replicas(routing: str) -> None:
    assert route_prompt_indices(5, 2, request_routing=routing) == ((0, 2, 4), (1, 3))


def test_route_prompt_indices_can_continue_round_robin_across_generation_rounds() -> None:
    assert route_prompt_indices(1, 2, starting_replica=0) == ((0,), ())
    assert route_prompt_indices(1, 2, starting_replica=1) == ((), (0,))


def test_replica_router_rejects_unknown_routing_policy() -> None:
    with pytest.raises(ValueError, match="unsupported request_routing"):
        route_prompt_indices(1, 2, request_routing="random")


def test_build_gpu_groups_rejects_non_divisible_shape() -> None:
    with pytest.raises(ValueError, match="requires 4 GPUs"):
        build_gpu_groups(gpu_count=2, tensor_parallel_size=2, replica_count=2)
