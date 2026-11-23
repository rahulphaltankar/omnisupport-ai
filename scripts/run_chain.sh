#!/usr/bin/env bash
# Sequential runner for the heavy experiment scripts (logs to logs/NN_name.out). Usage: bash scripts/run_chain.sh 04 05 ...
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for n in "$@"; do
  f=$(ls scripts/${n}_*.py | head -1)
  echo "== $f $(date +%T)" >> logs/chain.out
  python "$f" > "logs/${n}.out" 2>&1 || echo "FAILED $f" >> logs/chain.out
done
echo "chain done $(date +%T)" >> logs/chain.out
