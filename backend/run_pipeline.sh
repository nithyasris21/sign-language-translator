#!/bin/bash
# all downloaded WLASL videos + one reference clip per remaining word;
# already-processed words sit in data/raw_done
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
rm -f logs/DONE logs/FAILED
.venv/Scripts/python -m ml.preprocess > logs/preprocess_all.log 2>&1 \
 && mv data/raw_done/* data/raw/ && rmdir data/raw_done \
 && .venv/Scripts/python -m ml.train > logs/train.log 2>&1 \
 && grep -E "accuracy|baseline|samples across" logs/train.log > logs/DONE || touch logs/FAILED
