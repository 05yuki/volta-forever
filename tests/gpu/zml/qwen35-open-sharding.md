# examples/llm: Qwen3.5 fails to compile with `OpenShardingAtBoundaries` since #1004

What `zml-8898051c7b-close-open-outputs.diff` fixes, and why `build-zml.sh` is given that patch: written
as a report for the ZML project (https://github.com/zml/zml).

## Summary

At `8898051c` (main on 2026-11-04), `//examples/llm` cannot run a Qwen3.5 model: the first executable
the example compiles, `EmbedTokens.forward`, is rejected with `error.OpenShardingAtBoundaries`. It
happens before any weight is used, on the CPU platform as well as on CUDA.

#1004 (`4c355c61`, "zml/sharding: keep unconstrained graph results open") made an open dimension on an
executable output an error. `examples/llm/models/qwen3_5/model.zig` still returns tensors that carry
the `.model` mesh with only the `d` dimension constrained.

## To reproduce

```sh
git checkout 8898051c7b6b8cd015cdce8e75afc5404396405f
bazel run //examples/llm -- --model=<a Qwen3.5 checkpoint> --seqlen=128 --prompt="The capital of France is"
```

Seen with Qwen3.5-9B (`"model_type": "qwen3_5"`) on x86-64 Linux (Ubuntu 24.04), with the binary built
with and run without a GPU (platform `cpu`), and on a CUDA device. It takes two seconds to fail.

## Output

```
info(llm): Selected backend: .{ .stablehlo = void }
info(llm): Resolving model repository..
info(llm): Initializing model..
info(llm): Detected model type: .qwen3_5
info(llm): Loaded tokenizer [1.228s]
error(zml/Compiler): Executable output 0 has open sharding: {b=1/?,s=128/?,d=4096,f16}@model. Specify a logical mesh axis or .replicated for every dimension.
info(qwen3_5): Compiled prefill embed tokens [9.569ms]
error: OpenShardingAtBoundaries
```

## Cause

`examples/llm/models/qwen3_5/model.zig`, `EmbedTokens.forward`:

```zig
return .{ .hidden = input.embedding.embed_tokens.forward(tokens)
    .withPartialTags(.{.d})
    .withPartitioning(.model, .{ .d = .replicated }) };
```

The result has a mesh, `d` is replicated, `b` and `s` are left open. Since #1004,
`collectOutputInfo` in `zml/Compiler.zig` only defaults an output to replication when it has no mesh
at all:

```zig
const output = if (tensor.shape()._sharding.mesh != null or tensor.shape().isFullyReplicated()) tensor.* else tensor.replicate();
const og_shape = output.shape();
if (og_shape.containsPartitionSpec(.open)) {
    log.err("Executable output {d} has open sharding: {f}. ...", .{ ctx.infos.len, og_shape });
    return error.OpenShardingAtBoundaries;
}
```

`EmbedTokens` is only the first executable to hit it. The same file returns layer outputs constrained
the same way (`.withPartitioning(.model, .{ .d = .replicated })` on `hidden` in the prefill and decode
layers), so fixing that one function is not enough. `inference.zig` in the same directory already
constrains every dimension of what it returns
(`.withPartitioning(mesh, .{ .b = .replicated, .s = .replicated, .d = .replicated })`).

## Possible fixes

1. In the example: constrain every dimension of every tensor an executable returns, as
   `inference.zig` does. Not tried here.
2. In the compiler: when an output has a mesh and open dimensions, close them as replicated, which is
   what an output without a mesh already gets. With this change alone on top of `8898051c`, the
   example compiles its eight executables and generates text on the CPU platform (output below), and
   compiles them on CUDA:

```diff
--- a/zml/Compiler.zig
+++ b/zml/Compiler.zig
@@ -671,7 +671,14 @@
         fn cb(ctx: *@This(), tensor: *const Tensor) !void {
             // Outputs with no declared sharding default to replication. Constrain the
             // returned value so the partitioner produces a replicated output buffer.
-            const output = if (tensor.shape()._sharding.mesh != null or tensor.shape().isFullyReplicated()) tensor.* else tensor.replicate();
+            var output = if (tensor.shape()._sharding.mesh != null or tensor.shape().isFullyReplicated()) tensor.* else tensor.replicate();
+            if (output.shape()._sharding.mesh) |out_mesh| if (output.shape().containsPartitionSpec(.open)) {
+                var partitioning = output.shape()._sharding.partition;
+                for (0..output.rank()) |ax| {
+                    if (partitioning.get(ax) == .open) partitioning = partitioning.set(ax, .replicated);
+                }
+                output = output.withPartitioning(out_mesh, partitioning);
+            };
             const og_shape = output.shape();
             if (og_shape.containsPartitionSpec(.open)) {
                 log.err("Executable output {d} has open sharding: {f}. Specify a logical mesh axis or .replicated for every dimension.", .{ ctx.infos.len, og_shape });
```

   It gives up the strictness #1004 introduced at executable boundaries, so it may not be the fix you
   want; it is offered as a description of what makes the example run again.

   With the change, same command as above, CPU platform:

```
info(qwen3_5): Compiled prefill embed tokens [122.355ms]
info(qwen3_5): Compiled prefill full attention layer [216.95ms]
info(qwen3_5): Compiled prefill linear attention layer [239.976ms]
info(qwen3_5): Compiled prefill sampler [198.106ms]
info(qwen3_5): Compiled decode embed tokens [48.816ms]
info(qwen3_5): Compiled decode full attention layer [183.015ms]
info(qwen3_5): Compiled decode linear attention layer [248.213ms]
info(qwen3_5): Compiled decode sampler [195.346ms]
info(qwen3_5): Loaded weights [16.68GiB, 9.329s, 1.79GiB/s]
prefill...          <think>
Thinking Process:
1.  **Analyze the Request:** The user is asking a simple factual question: "The capital of France is". This is an incomplete sentence, implying they want me to complete it or provide the answer.
2.  **Identify the Fact:** The capital city of France is Paris.
[...]
decode 16m21.651s · 0.1tok/s
```

## Other models

Only Qwen3.5 was run. The expression `.withPartitioning(.model, .{ .d = .replicated })`, which leaves
the other dimensions open, is also in the other models of `examples/llm`: 7 times in
`llama/model.zig` (its `EmbedTokens.forward` returns the same expression as Qwen3.5's), 11 times in
`qwen3_5_moe/model.zig`, twice in `lfm2/model.zig`; `qwen3_5/model.zig` has it 15 times. Which of these
are returned by an executable was only checked for the two `EmbedTokens.forward`.
