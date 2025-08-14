#!/bin/bash
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

if [ ! -f server.log ]; then
  echo "No server.log found in $(pwd)"
  exit 1
fi

tail -n 200 -f server.log | cat


