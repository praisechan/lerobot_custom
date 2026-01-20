#!/bin/bash

TRACE_DIR="/tmp/pi0-libero-profile"

echo "=========================================="
echo "JAX Profile Traces Viewer Guide"
echo "=========================================="
echo ""

if [ ! -d "$TRACE_DIR" ]; then
    echo "❌ No traces found at $TRACE_DIR"
    echo "Run profile_pi0_detailed.py first to generate traces."
    exit 1
fi

TRACE_FILES=$(find "$TRACE_DIR" -name "*.pb" -o -name "*.json.gz" 2>/dev/null)

if [ -z "$TRACE_FILES" ]; then
    echo "❌ No trace files found in $TRACE_DIR"
    exit 1
fi

echo "✅ Found trace files:"
find "$TRACE_DIR" -name "*.pb" -o -name "*.json.gz" | while read file; do
    echo "   - $file"
done

echo ""
echo "=========================================="
echo "Option 1: View in Perfetto (Recommended)"
echo "=========================================="
echo ""
echo "Since you're on a remote machine (atlas2), download the trace to your local machine:"
echo ""
echo "  # On your local machine terminal:"
echo "  scp -r atlas2:$TRACE_DIR ./pi0-traces"
echo ""
echo "Then:"
echo "  1. Go to https://ui.perfetto.dev/ in your browser"
echo "  2. Click 'Open trace file'"
echo "  3. Select one of these files from ./pi0-traces/:"
echo "     - atlas2.trace.json.gz (best for viewing)"
echo "     - perfetto_trace.json.gz (alternative)"
echo ""

echo "=========================================="
echo "Option 2: View with TensorBoard + XProf"
echo "=========================================="
echo ""
echo "If you have TensorBoard installed with xprof plugin:"
echo ""
echo "  # Start TensorBoard"
echo "  tensorboard --logdir=$TRACE_DIR --port=6006"
echo ""
echo "  # Then in VSCode:"
echo "  1. Forward port 6006 (Ports tab → Forward Port)"
echo "  2. Open http://localhost:6006/ in your browser"
echo "  3. Click 'PROFILE' tab"
echo "  4. Select the trace from dropdown"
echo ""
echo "⚠️  Note: Currently blocked by protobuf version conflict"
echo "    Need protobuf >= 5.0 for xprof to work"
echo ""

echo "=========================================="
echo "Option 3: Python Perfetto Viewer"
echo "=========================================="
echo ""
echo "You can also view traces programmatically:"
echo ""
echo "  python -c '"
echo "import gzip, json"
echo "with gzip.open(\"$TRACE_DIR/plugins/profile/*/atlas2.trace.json.gz\", \"rt\") as f:"
echo "    trace = json.load(f)"
echo "    print(f\"Trace has {len(trace.get('traceEvents', []))} events\")"
echo "'"
echo ""

echo "=========================================="
echo "What to look for in the trace"
echo "=========================================="
echo ""
echo "Look for these custom TraceAnnotation markers:"
echo ""
echo "  📊 sample_actions (outer function)"
echo "    ├─ preprocess_observation"
echo "    ├─ embed_prefix"
echo "    ├─ compute_kv_cache"
echo "    └─ diffusion_loop (10 iterations)"
echo "        ├─ step_embed_suffix"
echo "        ├─ step_compute_masks"
echo "        ├─ step_compute_positions"
echo "        ├─ step_llm_forward ⭐ (main bottleneck)"
echo "        └─ step_action_proj"
echo ""
echo "The step_* markers work inside JIT-compiled code!"
echo ""
