#!/bin/bash
# Master script to run both generation batches sequentially

echo "🚀 Starting sequential batch generation at $(date)"

# Run first batch directly (no nohup/background since scripts are now clean)
echo "▶️  Starting first batch: run_weighted_generation_fixed.sh"
./run_weighted_generation_fixed.sh > batch1_output.log 2>&1
BATCH1_EXIT_CODE=$?

if [ $BATCH1_EXIT_CODE -eq 0 ]; then
    echo "✅ First batch completed successfully at $(date)"
    
    # Run second batch directly
    echo "▶️  Starting second batch: run_weighted_generation_fixed_v2.sh" 
    ./run_weighted_generation_fixed_v2.sh > batch2_output.log 2>&1
    BATCH2_EXIT_CODE=$?
    
    if [ $BATCH2_EXIT_CODE -eq 0 ]; then
        echo "🎉 Both batches completed successfully at $(date)"
    else
        echo "❌ Second batch failed at $(date)"
        exit 1
    fi
else
    echo "❌ First batch failed at $(date), skipping second batch"
    exit 1
fi

echo "📊 All generation batches completed at $(date)"
