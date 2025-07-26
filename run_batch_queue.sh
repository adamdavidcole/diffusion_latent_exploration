#!/bin/bash
# Queue multiple generation batches to run sequentially

echo "🚀 Starting batch queue at $(date)"

# Batch 1: Kiss variations
echo "▶️  Starting Batch 1: Kiss variations with weight 2.5"
python main.py --config configs/wan_14b_optimized_fixed.yaml --device cuda:1 --template "a romantic kiss between [two (gay:2.5) men | two (men:2.5) | two people | two (women:2.5) | two (lesbian women:2.5) | a (man:2.5) and a (man:2.5) | a (woman:2.5) and a (woman:2.5) | two (black:2.5) people | two (white:2.5) people | two (hispanic:2.5) people | two (asian:2.5) people]" --videos-per-variation 4 --batch-name "14b_kiss_2.5frame_weighted"

if [ $? -eq 0 ]; then
    echo "✅ Batch 1 completed successfully at $(date)"
else
    echo "❌ Batch 1 failed at $(date)"
    exit 1
fi

# Batch 2: Different template or settings
echo "▶️  Starting Batch 2: Kiss variations with weight 5.0"
python main.py --config configs/wan_14b_optimized_fixed.yaml --device cuda:1 --template "a romantic kiss between [two (gay:5.0) men | two (men:5.0) | two people | two (women:5.0) | two (lesbian women:5.0)]" --videos-per-variation 4 --batch-name "14b_kiss_5.0frame_weighted"

if [ $? -eq 0 ]; then
    echo "✅ Batch 2 completed successfully at $(date)"
else
    echo "❌ Batch 2 failed at $(date)"
    exit 1
fi

# Batch 3: Add more batches as needed
echo "▶️  Starting Batch 3: Dancing variations"
python main.py --config configs/wan_14b_optimized_fixed.yaml --device cuda:1 --template "a person (dancing:3.0) in [the park | the street | a club | at home]" --videos-per-variation 3 --batch-name "14b_dancing_3.0frame_weighted"

if [ $? -eq 0 ]; then
    echo "✅ Batch 3 completed successfully at $(date)"
else
    echo "❌ Batch 3 failed at $(date)"
    exit 1
fi

echo "🎉 All batches completed successfully at $(date)"

# Optional: Send notification, cleanup, or analysis
echo "📊 Running post-processing..."
# Add any cleanup or analysis commands here

echo "📧 Sending completion notification..."
# Add notification command if you have one set up
