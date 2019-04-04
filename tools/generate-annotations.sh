#!/bin/bash
CDIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"

PYTHON=python3.6
#TARDIR=$CDIR/annotations
TARDIR=`pwd`/annotations
SRC_PIPELINES_DIR=/validation_output

primitives=$(python3.6 -m d3m.index search 2>/dev/null| grep d3m.primitives.bbn)
if [ $? -ne 0 ]; then
  echo "An error occured while getting the list of primitives" > /dev/stderr
  exit 1
fi

pipelines=$(find $SRC_PIPELINES_DIR/ -maxdepth 1 | grep -e "pipeline.*\.json$")

pipeline_meta=`mktemp`
cat << EOF > $pipeline_meta
{
   "problem": "31_urbansound_problem",
   "train_inputs": ["31_ud_dataset_TRAIN"],
   "test_inputs": ["31_ud_dataset_TEST"]
}
EOF

for primitive in $primitives; do
  json=`mktemp`
  stderr=`mktemp`
  #echo "python3.6 -m d3m.index -s $primitive 2>/dev/null"
  python3.6 -m d3m.index describe $primitive 2>$stderr \
    > $json
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate annotation for primitive $primitive" > /dev/stderr
    cat $stderr > /dev/stderr
    if [ $(echo -e $stderr | grep "Primitive to register has to be a class." | wc -l) -ge 0 ]; then
      echo "IGNORED" > /dev/stderr
      continue
    else
      exit 1
    fi
  fi

  python_path=$($PYTHON $CDIR/parse-json.py $json 'python_path')
  version=$($PYTHON $CDIR/parse-json.py $json 'version')
  primitive_id=$($PYTHON $CDIR/parse-json.py $json 'id')
  #cat primitive.json | grep "^\s\+\"id\":" | sed 's@^.*"id":[^"]*"\([^"]\+\)".*$@\1@'
  ATARDIR=$TARDIR/$python_path/$version
  mkdir -p $ATARDIR
  # copy annotation
  mv $json $ATARDIR/primitive.json
  # copy pipelines using the primitive
	for pipeline in $pipelines; do
		pipeline_id=$($PYTHON $CDIR/parse-json.py $pipeline 'id')
		if [ $(grep -i "$primitive_id" $pipeline | wc -l) -ne 0 ]; then
      mkdir -p $ATARDIR/pipelines
      cp $pipeline $ATARDIR/pipelines/$pipeline_id.json
      cp $pipeline_meta $ATARDIR/pipelines/$pipeline_id.meta
		fi
  done

  rm $stderr
done

chmod -R 777 $TARDIR
