#!/bin/bash

# Group: 14
# Date: 19 April 2026
# Members: Deodato V. Bastos Neto, Karina Musina

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "🤖 Starting Robot Mission Multi-Agent Systems Experiments "
echo "=========================================================="
echo "This script will run several batch experiments to analyze"
echo "the impact of different agent features on the time to clear."

# Global Settings
PROCESSES=8       # Number of CPU cores to use (adjust if needed)
ITERATIONS=10     # Number of simulation runs per parameter combination
MAX_STEPS=1500    # Timeout for a single run

echo ""
echo "📊 Experiment 1: The Impact of Communication"
echo "Testing how the Peer-to-Peer network and dead-lock negotiation affects performance."
python batch_experiments.py \
    --use-communication True,False \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_communication

echo ""
echo "📊 Experiment 2: The Impact of Green-to-Green Coordination"
echo "Testing whether local negociation about pickup between visible green robots improves throughput."
python batch_experiments.py \
    --green-coordination-values True,False \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_green_coordination

echo ""
echo "📊 Experiment 3: The Impact of Red Agent Memory"
echo "Testing how remembering the disposal zone and dropped wastes affects performance."
python batch_experiments.py \
    --use-memory True,False \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_memory

echo ""
echo "📊 Experiment 4: The Impact of Border Patrol"
echo "Testing if proactive border patrolling reduces waiting time for wastes."
python batch_experiments.py \
    --patrol-border True,False \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_patrol

echo ""
echo "📊 Experiment 5: Initial Waste Distribution"
echo "Testing performance difference between starting with only Green wastes vs Pre-distributed wastes."
python batch_experiments.py \
    --multiple-wastes True,False \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_multiple_wastes

echo ""
echo "📊 Experiment 6: Quantitative Scaling (Robots, Vision & Waste Count)"
echo "Testing the model's scalability by varying continuous parameters."
# Note: Lowering iterations here because combinations multiply quickly!
python batch_experiments.py \
    --design ofat \
    --n-waste 4,16,24,32,40,48,56  \
    --n-green-robots 1,2,3,4,6,8 \
    --n-yellow-robots 1,2,3,4,5 \
    --n-red-robots 1,2,3,4,5 \
    --vision 1,2,3,4 \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_scaling

echo ""
echo "📊 Experiment 7: Vision as a Communication Fallback"
echo "Testing if a large vision radius can compensate for having no communication."
python batch_experiments.py \
    --vision 1,3,5 \
    --use-communication False \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_vision_no_comm

echo ""
echo "📊 Experiment 8: Extreme Crowding & Deadlock Stress Test"
echo "Testing how the system handles severe traffic jams with many agents and little waste."
python batch_experiments.py \
    --n-green-robots 10,15 \
    --n-yellow-robots 8,12 \
    --n-waste 16 \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_extreme_crowding

echo ""
echo "📊 Experiment 9: The 'Lone Wolf' Baseline"
echo "Testing efficiency when exactly one robot of each type must clear the entire grid."
python batch_experiments.py \
    --n-green-robots 1 \
    --n-yellow-robots 1 \
    --n-red-robots 1 \
    --n-waste 16,32,64 \
    --iterations $ITERATIONS \
    --max-steps $MAX_STEPS \
    --processes $PROCESSES \
    --outdir batch_results/exp_lone_wolf

echo ""
echo "✅ All experiments completed successfully!"
echo "Check the 'batch_results' subdirectories for CSV files and bar/line charts."
