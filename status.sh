#!/bin/bash
# Where the Stage 2h regeneration chain actually is. Reads the process table
# and the files on disk, not anything Claude wrote down.
SCR=/tmp/claude-464589609/-Users-martin-torres-Library-CloudStorage-Dropbox-Work-CUBoulder-Dissertation-Coding-CompareUQMethods/c603a482-0912-4f65-800d-c5bd9df24d56/scratchpad
cd "$(dirname "$0")"
echo "=== $(date '+%H:%M:%S') ==="
if pgrep -f "chain.sh" >/dev/null; then
  echo "CHAIN: running (pid $(pgrep -f chain.sh | head -1))"
  echo "  current step: $(grep '^=== ' $SCR/chain.log 2>/dev/null | tail -1)"
  echo "  python:       $(ps -o etime=,command= -p $(pgrep -f '[p]ython' | head -1) 2>/dev/null | cut -c1-90)"
else
  echo "CHAIN: NOT RUNNING"
  echo "  last step logged: $(grep '^=== ' $SCR/chain.log 2>/dev/null | tail -1)"
fi
echo
echo "active corpus: $(python3 -c "import json;print(json.load(open('data/processed/CORPUS.json'))['active_corpus'])" 2>/dev/null)"
echo "corpus files:  $(ls data/processed/corpus_2026-09-24/ 2>/dev/null | tr '\n' ' ')"
echo
echo "outputs written in the last 30 min:"
find outputs/tables outputs/figures -newermt '30 minutes ago' -type f 2>/dev/null | head -8 | sed 's/^/  /'
[ -z "$(find outputs/tables -newermt '30 minutes ago' -type f 2>/dev/null)" ] && echo "  (none)"
exit 0
