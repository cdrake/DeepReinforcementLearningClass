#!/bin/bash

# Helper script to run homework assignments
# Automatically activates the rlclass conda environment

# Initialize conda
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate rlclass
echo "Using environment: $CONDA_DEFAULT_ENV"
echo ""

usage() {
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  mrp                  Run MRP task (HW1)"
    echo "  dp [env]             Run Dynamic Programming (HW1)"
    echo "  mf [env]             Run Model-Free RL (HW1)"
    echo "  hw2 [--rand]         Run Deep RL training (HW2)"
    echo "  hw3 [--grade]        Run Search with DQN (HW3)"
    echo ""
    echo "Environments for dp/mf:"
    echo "  aifarm               Deterministic farm (default)"
    echo "  aifarm_0.2           Stochastic farm (20% random right)"
    echo ""
    echo "Options:"
    echo "  --grade              Grade against expected output"
    echo "  --gamma <value>      Set discount factor (default: 0.9)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh mrp"
    echo "  ./run.sh dp aifarm"
    echo "  ./run.sh dp aifarm_0.2 --grade"
    echo "  ./run.sh mf aifarm --gamma 0.95"
    echo "  ./run.sh hw2 --rand"
    echo "  ./run.sh hw3 --grade"
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

COMMAND=$1
shift

case $COMMAND in
    mrp)
        GAMMA="0.9"
        while [[ $# -gt 0 ]]; do
            case $1 in
                --gamma) GAMMA="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        python run_hw1.py --task mrp --gamma "$GAMMA"
        ;;
    dp)
        ENV="${1:-aifarm}"
        shift 2>/dev/null
        GAMMA="0.9"
        GRADE=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --gamma) GAMMA="$2"; shift 2 ;;
                --grade) GRADE="--grade"; shift ;;
                *) shift ;;
            esac
        done
        python run_hw1.py --task dp --env "$ENV" --gamma "$GAMMA" $GRADE
        ;;
    mf)
        ENV="${1:-aifarm}"
        shift 2>/dev/null
        GAMMA="0.9"
        GRADE=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --gamma) GAMMA="$2"; shift 2 ;;
                --grade) GRADE="--grade"; shift ;;
                *) shift ;;
            esac
        done
        python run_hw1.py --task mf --env "$ENV" --gamma "$GAMMA" $GRADE
        ;;
    hw2)
        RAND=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --rand) RAND="--rand"; shift ;;
                *) shift ;;
            esac
        done
        python run_hw2.py $RAND
        ;;
    hw3)
        GRADE=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --grade) GRADE="--grade"; shift ;;
                *) shift ;;
            esac
        done
        python run_hw3.py $GRADE
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
