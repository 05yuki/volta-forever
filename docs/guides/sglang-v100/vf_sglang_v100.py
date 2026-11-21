def _sum_experts(x, out, routed_scaling_factor):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import moe_sum_reduce_torch_compile
    return moe_sum_reduce_torch_compile(x, out, routed_scaling_factor)


def _align_experts(original, topk_ids, num_experts, block_size, sorted_ids, expert_ids,
                   num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids=False,
                   ignore_invalid_expert=False):
    args = (topk_ids, num_experts, block_size, sorted_ids, expert_ids,
            num_tokens_post_pad, cumsum_buffer, pad_sorted_token_ids)
    if ignore_invalid_expert:
        return original(*args, ignore_invalid_expert=True)
    from sglang.kernels.ops.moe.moe_align import moe_align_block_size
    return moe_align_block_size(*args)


def configure():
    from torch.backends import cuda
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    cuda.enable_flash_sdp(False)
    cuda.enable_mem_efficient_sdp(False)
    cuda.enable_cudnn_sdp(False)
    cuda.enable_math_sdp(True)
    HookRegistry.register(
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.moe_sum_reduce",
        _sum_experts,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.moe_sum_reduce",
        _sum_experts,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size.sgl_moe_align_block_size",
        _align_experts,
        HookType.AROUND,
    )
    print("SGLang: selected PyTorch MATH SDPA and SGLang's PyTorch expert sum for Volta", flush=True)
