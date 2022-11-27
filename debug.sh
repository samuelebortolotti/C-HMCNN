#!/bin/sh

# Directives to SLURM

#SBATCH -p chaos 
#SBATCH -A shared-sml-staff
#SBATCH --signal=B:SIGTERM@120
#SBATCH --gres gpu:1
#SBATCH -t 20:00:00
#SBATCH --mem=10G
#SBATCH -o /nfs/data_chaos/sbortolotti/logs/C-HMCNN/debug_resnet.out
#SBATCH -e /nfs/data_chaos/sbortolotti/logs/C-HMCNN/debug_resnet.e

# SCRIPT: run.sh
# AUTHOR: Samuele Bortolotti <samuele@studenti.unitn.it>
# DATE:   2022-14-11
#
# PURPOSE: Runs the network debugger

usage() {
  test $# = 0 || echo "$@"
  echo
  echo Trains and evaluates the network on the gpu cluster
  echo Options:
  echo "  -h, --help                    Print this help"
  echo
  exit 1
}

args=
while [ $# != 0 ]; do
  case $1 in
    -h|--help) usage ;;
    -?*) usage "Unknown option: $1" ;;
    *) args="$args \"$1\"" ;;
  esac
  shift
done

# Get args
eval "set -- $args"

# enter in the code directory
cd "/nfs/data_chaos/sbortolotti/code/C-HMCNN"

# load the right python environment
python="/nfs/data_chaos/sbortolotti/pkgs/miniconda/envs/chmncc/bin/python"

# load wandb
wandb="/nfs/data_chaos/sbortolotti/pkgs/miniconda/envs/chmncc/bin/wandb"

# log in wandb (REQUIRES KEY)
$wandb login $KEY

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

# run the experiment
${python} -m chmncc debug --learning-rate 0.001 --batch-size 128 --test-batch-size 256 --device "cuda" --project chmncc --wandb true --network "resnet"


wait
