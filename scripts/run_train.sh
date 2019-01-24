#!/bin/bash
# This script trains a model.
# Usage: run_train.sh [-cy] [-g gpu] [exp_dir]
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

function usage {
  echo "Usage: run_train.sh [-cy] [-g gpu] [exp_dir]"
  echo "Options:"
  echo "  -c        clear previous outputs and rerun"
  echo "  -g gpu    use a specific gpu (default to 0)"
  echo "  -h        display help"
  echo "  -y        automatic yes to prompts"
  exit 1
}

function run {
  for run_dir in $EXP_DIR;
  do
    if [ -f "$run_dir/config.yaml" ]
    then
      if $CLEAR
      then
        for subdir in "logs" "model" "src"
        do
          if [ -d "$run_dir/$subdir" ]
          then
            rm -r "$run_dir/$subdir"
          fi
        done
      fi
      if [ ! -d "$run_dir/logs/train" ]
      then
        python3 "$DIR/../src/train.py" --exp_dir "$run_dir" \
          --config "$run_dir/config.yaml" --gpu "$GPU"
      fi
    fi
  done
}

if [[ $# -eq 0 ]]
then
  usage
fi

EXP_DIR="${@: -1}"
CLEAR=false
GPU="0"
YES=false
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    -c|--clear)
      CLEAR=true
      shift
      ;;
    -g|--gpu)
      GPU="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    -y|--yes)
      YES=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

echo "Experiment summary:"
echo "  exp_dir: $EXP_DIR"
echo "  clear: $CLEAR"
echo "  gpu: $GPU"
echo "  runs:"

if [ ! -f "$EXP_DIR/config.yaml" ]
then
  EXP_DIR="$EXP_DIR/*/"
fi

for run_dir in $EXP_DIR
do
  if [ -f "$run_dir/config.yaml" ]
  then
    if $CLEAR
    then
      echo "    $run_dir"
    elif [ ! -d "$run_dir/logs/train" ]
    then
      echo "    $run_dir"
    fi
  fi
done

if $YES
then
  run
else
  read -p "Are you sure to start the experiment? " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]
  then
    run
  else
    echo "Experiment cancelled."
  fi
fi
