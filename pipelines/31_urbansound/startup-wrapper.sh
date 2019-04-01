#!/bin/bash
set -eo pipefail
CDIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"

INPUT_DIR=$1
OUTPUT_DIR=$2
BBN_PIPELINE=$3
#TA1_RUNTIME=$4
TA1_RUNTIME=$CDIR/ta1-pipeline

mkdir -p $OUTPUT_DIR

TMP=`mktemp`
cat $CDIR/startup.sh \
  | sed 's@^INPUT_DIR=.*$@INPUT_DIR='$INPUT_DIR'@' \
  | sed 's@^OUTPUT_DIR=.*$@OUTPUT_DIR='$OUTPUT_DIR'@' \
  | sed 's@^TA1_RUNTIME=.*$@TA1_RUNTIME='$TA1_RUNTIME'@' \
  > $TMP
#  | sed 's@^BBN_PIPELINE=.*$@BBN_PIPELINE='$BBN_PIPELINE'@' \

chmod a+x $TMP
export BBN_PIPELINE=$BBN_PIPELINE
cat $TMP

$TMP

