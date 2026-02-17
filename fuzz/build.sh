#!/usr/bin/env bash
set -e
rm -rf binaries
mkdir binaries
MOUNT_PATH="/binaries"


echo "[*] Building fuzz target with AFL++"


# Try with curl for now
# Adapted from Bosch Research's MLFuzz: https://github.com/boschresearch/mlfuzz/blob/main/benchmarks/build_fuzzbench_targets.py
BASE="$(dirname "$0")/targets/curl_curl_fuzzer_http"
SRC="$BASE/curl"
WORK="$BASE/b"
CC=afl-clang-fast
CFLAGS="-O2 -g"

echo "Build ${SRC} to ${MOUNT_PATH}"
cd "$BASE" && git clone https://github.com/curl/curl-fuzzer && \
git -C ./curl-fuzzer checkout 1fa81171ccf929ecb10f23256710b85b67f93e0d && \
git clone  https://github.com/curl/curl.git && \
git -C ./curl checkout 2eebc58c4b8d68c98c8344381a9f6df4cca838fd

mkdir ${WORK} && mkdir /src && cd ${BASE}/curl-fuzzer && chmod a+x ./ossfuzz.sh && \
sed -i 's#install_curl.sh /src/curl#install_curl.sh ${SRC}#' ./ossfuzz.sh && ./ossfuzz.sh

cd ${MOUNT_PATH} && mv curl_fuzzer_http curl.neuzzpp_torch && mv http.dict ./dicts/curl.dict && \
unzip -ou curl_fuzzer_http_seed_corpus.zip -d ./seeds/curl && \
rm curl_fuzzer* fuzz_url*


echo "[+] Build complete: ${SRC}"
