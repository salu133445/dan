#!/bin/bash
# This script set ups a new exeperiment.
# Usage: setup_exp.sh [-n note] [-r runs] [exp_dir]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

function usage {
  echo "Usage: setup_exp.sh [-n note] [-r runs] [exp_dir]"
  echo "Options:"
  echo "  -h         display help"
  echo "  -n note    experiment note"
  echo "  -r runs    number of runs"
  exit 1
}

if [[ $# -eq 0 ]]
then
  usage
fi

EXP_DIR="${@: -1}"
NOTE=$EXP_DIR
RUNS=1
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -h|--help)
      usage
      ;;
    -n|--note)
      NOTE="$2"
      shift 2
      ;;
    -r|--runs)
      RUNS="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p "$EXP_DIR"
for (( i=0; i<$RUNS; i++ ))
do
  run_dir="$EXP_DIR/run$i"
  mkdir -p "$run_dir"
  error=$(cp "$DIR/../src/dan/default_config.yaml" "$run_dir/config.yaml")
  if [[ $error ]]
  then
    echo $error
    exit 1
  fi
done

echo "$NOTE" > "$EXP_DIR/exp_note.txt"

echo "Experiment successfully created."
echo "  exp_dir: $EXP_DIR"
echo "  exp_note: $NOTE"
echo "  runs: $RUNS"
