#!/usr/bin/env bash
# After the heavy chain: NLP classical, then the (latency-sensitive) LLM benchmark, then the monitoring workload.
cd "$(dirname "$0")/.."
export PYTHONPATH=src
until grep -q "chain2 done" logs/chain2.out 2>/dev/null; do sleep 30; done
for n in 08 12 16; do
  f=$(ls scripts/${n}_*.py | head -1)
  echo "== $f $(date +%T)" >> logs/chain3.out
  python "$f" > "logs/${n}.out" 2>&1 || echo "FAILED $f" >> logs/chain3.out
done
echo "chain3 done $(date +%T)" >> logs/chain3.out
