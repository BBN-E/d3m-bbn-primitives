#!/bin/bash
set -eo pipefail
CDIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"

INPUT_DIR=/input
OUTPUT_DIR=/output
#TA1_RUNTIME=/user_opt/ta1-pipeline
TA1_RUNTIME=$CDIR/ta1-pipeline
#BBN_PIPELINE=$(find $CDIR/ | grep -e "pipeline.*json$" | head -1)

if [ "x$BBN_PIPELINE" == "x" ]; then
  echo "No pipeline available" > /dev/stderr
  exit 1
fi

#clean up the output area
cd $OUTPUT_DIR
rm -rf *

#require directories for submission
PIPELINES=$OUTPUT_DIR/pipelines
EXECUTABLES=$OUTPUT_DIR/executables
SUPPORTING_FILES=$OUTPUT_DIR/supporting_files
PREDICTIONS=$OUTPUT_DIR/predictions
#workdir
WORKDIR=$OUTPUT_DIR/workdir
OUT_PIPELINE_ID="d3m_$(basename $BBN_PIPELINE | sed 's@.json$@@')"

for d in $PIPELINES $EXECUTABLES $SUPPORTING_FILES $PREDICTIONS $WORKDIR; do
  mkdir -p $d
done

cat << EOF > $WORKDIR/ta1-pipeline-config.json
{
  "train_data":"$INPUT_DIR/TRAIN",
  "test_data":"$INPUT_DIR/TEST",
  "output_folder":"$PREDICTIONS/$OUT_PIPELINE_ID"
}
EOF

#execute ta1-pipeline code to generate submission product
OUT_D3M_PIPELINE=$PIPELINES/$OUT_PIPELINE_ID.json
python3.6 $TA1_RUNTIME \
  --ta1_config $WORKDIR/ta1-pipeline-config.json \
  --pipeline $BBN_PIPELINE \
  --d3m_pipeline_outfn $OUT_D3M_PIPELINE

D3M_PIPELINE_ID=$(python3.6 $CDIR/parse-json.py $OUT_D3M_PIPELINE 'id')
OUT_D3M_PIPELINE2="$(dirname $OUT_D3M_PIPELINE)/$D3M_PIPELINE_ID.json"
mv $OUT_D3M_PIPELINE $OUT_D3M_PIPELINE2

OUT_PIPELINE_ID=$D3M_PIPELINE_ID
mkdir -p $SUPPORTING_FILES/$OUT_PIPELINE_ID $PREDICTIONS/$OUT_PIPELINE_ID

OUT_SUPPORTING_FILES=$SUPPORTING_FILES/$OUT_PIPELINE_ID
OUT_EXECUTABLE=$EXECUTABLES/$OUT_PIPELINE_ID.sh
cp $TA1_RUNTIME $SUPPORTING_FILES/$OUT_PIPELINE_ID/
cp $BBN_PIPELINE $SUPPORTING_FILES/$OUT_PIPELINE_ID/

cat << EOF > $OUT_EXECUTABLE
#!/bin/bash
CDIR="\$( cd "\$(dirname "\${BASH_SOURCE[0]}" )" && pwd )"

python3.6 \$CDIR/../supporting_files/$OUT_PIPELINE_ID/ta1-pipeline \\
  --ta1_config \$1 \\
  --pipeline \$CDIR/../supporting_files/$OUT_PIPELINE_ID/$(basename $BBN_PIPELINE) \\
  --disable_d3m_pipeline_creation
EOF

chmod a+x $OUT_EXECUTABLE
