#!/usr/bin/env bash
# Download the Liu et al. (2020) CME dataset from the authors' public repository
# (MIT license), pinned to the commit used in this study, and copy the files used here.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=https://github.com/deepsuncode/RNN-CME-prediction.git
COMMIT=ee455ebeccfd95939d6c099267517c1f5bfe33a8
TMP=$(mktemp -d)
git clone --quiet "$REPO" "$TMP/RNN-CME-prediction"
git -C "$TMP/RNN-CME-prediction" checkout --quiet "$COMMIT"
mkdir -p data
cp "$TMP/RNN-CME-prediction/CME_data_samples/normalized_training_24.csv" data/
cp "$TMP/RNN-CME-prediction/CME_data_samples/normalized_testing_24.csv" data/
cp "$TMP/RNN-CME-prediction/CMEpredict/normalized_testing_24.csv" data/normalized_testing_24_demo540.csv
cp "$TMP/RNN-CME-prediction/CMEpredict/normalized_training_24.csv" data/codefolder_normalized_training_24.csv
cp "$TMP/RNN-CME-prediction/CMEpredict/lstm-24-output.csv" data/lab_lstm-24-output.csv
rm -rf "$TMP"
echo "Data ready in ./data"
