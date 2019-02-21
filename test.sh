#!/bin/bash
set -eux

for sample in $PWD/*/*; do
  if [[ ${PZC_TARGET_ARCH} == "sc1-64" && $(basename $sample) == "Atomic" ]]; then
    continue
  fi
  cd $sample
  make
  make run
done
