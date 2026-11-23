#!/usr/bin/env bash
# Wait for the first heavy chain (regression, NN, vision), then run NLP, fine-tune, pretrained pipelines sequentially.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for i in $(seq 1 360); do
  if grep -q "regression complete" logs/04.out 2>/dev/null && grep -q "vision complete" logs/07_vis.out 2>/dev/null; then break; fi
  sleep 30
done
for n in 08 10 11; do
  f=$(ls scripts/${n}_*.py | head -1)
  echo "== $f $(date +%T)" >> logs/chain2.out
  python "$f" > "logs/${n}.out" 2>&1 || echo "FAILED $f" >> logs/chain2.out
done
echo "chain2 done $(date +%T)" >> logs/chain2.out
