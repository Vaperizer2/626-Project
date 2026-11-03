#!/usr/bin/env bash
# Every claim in 626-artifacts/trtbugs/NOTES.md, rerun from scratch on carlpc2 (any GPU with ORT 1.31.1 + TRT 10.16.1).
#   evidence.sh <work_dir>
# <work_dir> holds py (python with onnxruntime-gpu and the tensorrt bindings), tools/ (this directory),
# inputs/{384,308}_0.9000.npy (dev images, tools/tokens/build.py --inputs) and src/ with reflinked copies of
# baseline_fp32 and compose_int8_tome308. GPU steps take /tmp/626-gpu.lock.
set -uo pipefail
cd "$1"
P="./py tools/probe.py"
G="flock /tmp/626-gpu.lock"
INT8='{"trt_fp16_enable": true, "trt_int8_enable": true, "trt_max_workspace_size": 2147483648}'
off() { echo "{\"optimization.disable_specified_optimizers\": \"$1\"}"; }
show() { grep '^{' | python3 -c '
import json, sys
for l in sys.stdin:
    d = json.loads(l)
    if "cos_trt_vs_cuda" in d:
        print("   ", d["cos_trt_vs_cuda"])
    else:
        print("    batch %d: cos min %.4f rows %s" % (d["batch"], d["cos_min"], d["cos_rows"]))'; }
mkdir -p opt pure

B=src/baseline_fp32/model.onnx X=inputs/384_0.9000.npy
echo "## bug 1: baseline_fp32 (fp32 TensorRT, profile 1..8), TRT rows vs CUDA fp32 rows"
echo "- ORT_ENABLE_ALL";                               $G $P run $B $X | show
echo "- ORT_ENABLE_ALL, TransposeOptimizer disabled";   $G $P run $B $X --config "$(off TransposeOptimizer)" | show
echo "- ORT_ENABLE_ALL, ConstantFolding disabled";      $G $P run $B $X --config "$(off ConstantFolding)" | show
echo "- ORT_DISABLE_ALL";                              $G $P run $B $X --level ORT_DISABLE_ALL | show
echo "- ORT_ENABLE_ALL, static batch-8 engine";         $G $P run $B $X --batches 8 --profile static | show
echo "- separator: 'ConstantSharing,TransposeOptimizer' (comma)"
$G $P run $B $X --batches 2 --config "$(off ConstantSharing,TransposeOptimizer)" | show
echo "- separator: 'ConstantSharing;TransposeOptimizer' (semicolon)"
$G $P run $B $X --batches 2 --config "$(off 'ConstantSharing;TransposeOptimizer')" | show
$P level1 $B opt/baseline_level1.onnx
echo "- plain TensorRT (no onnxruntime in the build) on the exported graph, batch 1 then 2"
$G ./py tools/puretrt.py $B $X pure/baseline --batch 2 | show
echo "- plain TensorRT on the graph after onnxruntime's level-1 transformers"
$G ./py tools/puretrt.py opt/baseline_level1.onnx $X pure/baseline_level1 --batch 2 | show
$P level1 $B opt/baseline_level1_noto.onnx --disable TransposeOptimizer
echo "- plain TensorRT on the level-1 graph without TransposeOptimizer"
$G ./py tools/puretrt.py opt/baseline_level1_noto.onnx $X pure/baseline_level1_noto --batch 2 | show

T=src/compose_int8_tome308/model.onnx X=inputs/308_0.9000.npy
echo "## bug 2: compose_int8_tome308 (static batch 1, TensorRT fp16+int8), TRT rows vs CUDA fp32 (QDQ simulated)"
for cfg in "ORT_ENABLE_ALL:" "ORT_DISABLE_ALL:" "ORT_ENABLE_ALL:TransposeOptimizer" "ORT_ENABLE_ALL:ConstantFolding" \
           "ORT_ENABLE_ALL:MatMulAddFusion" "ORT_ENABLE_ALL:MatMulAddFusion;TransposeOptimizer"; do
  echo "- ${cfg%%:*}, disabled: ${cfg#*:}"
  $G $P run $T $X --batches 1 --rows 6 --trt "$INT8" --level "${cfg%%:*}" --config "$(off "${cfg#*:}")" | show
done
$P level1 $T opt/tome_int8_level1.onnx
echo "- plain TensorRT on the exported graph"
$G ./py tools/puretrt.py $T $X pure/tome_int8 --batch 1 --fp16 --int8 | show
echo "- plain TensorRT on the level-1 graph"
$G ./py tools/puretrt.py opt/tome_int8_level1.onnx $X pure/tome_int8_level1 --batch 1 --fp16 --int8 | show
rev() {
  ./py tools/revert_gemm.py opt/tome_int8_level1.onnx "opt/rev_$1.onnx" --roles "$2" > /dev/null
  echo "- level-1 graph, Gemm form kept only for: $3"
  $G $P run "opt/rev_$1.onnx" $X --batches 1 --rows 6 --trt "$INT8" --level ORT_DISABLE_ALL | show
}
all_but() { local s=attn_pool; for i in $(seq 0 26); do for r in attn__qkv attn__proj mlp__fc1 mlp__fc2; do
  [ "blocks__${i}__${r}" = "$1" ] || s="$s,blocks__${i}__${r}__"; done; done; echo "$s"; }
rev none "blocks__,attn_pool" "none (all 112 GEMMs back to MatMul + Add)"
rev fc1_b21 "$(all_but blocks__21__mlp__fc1)" "block 21 mlp.fc1 (256 tokens)"
rev fc1_b10 "$(all_but blocks__10__mlp__fc1)" "block 10 mlp.fc1 (484 tokens)"
rev qkv_b21 "$(all_but blocks__21__attn__qkv)" "block 21 attn.qkv (256 tokens)"
echo done
