#!/bin/bash
# Test script for comparison-based classification

set -e  # Exit on error

echo "=========================================="
echo "Comparison vs Selection Test"
echo "COLING 2025 Paper Implementation"
echo "=========================================="
echo ""

# Check if test data exists
if [ ! -f "data/test_notes.csv" ]; then
    echo "❌ Error: data/test_notes.csv not found"
    echo "Please provide test data first"
    exit 1
fi

# Create output directories
mkdir -p outputs/comparison
mkdir -p outputs/original

echo "🧪 Test 1: Original Selection Method"
echo "=========================================="
python main.py \
    --notes-path data/test_notes.csv \
    --submission-path outputs/original/submission.csv \
    --use-comparison-method False \
    2>&1 | tee outputs/original/log.txt

echo ""
echo "✅ Original method completed"
echo ""

echo "🧪 Test 2: Comparison Method (COLING 2025)"
echo "=========================================="
python main.py \
    --notes-path data/test_notes.csv \
    --submission-path outputs/comparison/submission.csv \
    --use-comparison-method True \
    2>&1 | tee outputs/comparison/log.txt

echo ""
echo "✅ Comparison method completed"
echo ""

echo "=========================================="
echo "📊 Results Comparison"
echo "=========================================="

# Count predictions
ORIGINAL_COUNT=$(wc -l < outputs/original/submission.csv)
COMPARISON_COUNT=$(wc -l < outputs/comparison/submission.csv)

echo "Original method predictions: $ORIGINAL_COUNT"
echo "Comparison method predictions: $COMPARISON_COUNT"

# Check if ground truth exists for scoring
if [ -f "data/test_annotations.csv" ]; then
    echo ""
    echo "🎯 Computing IoU scores..."

    # This would require implementing a scoring script
    # For now, just show the outputs
    echo "Ground truth found. Manual scoring required."
    echo "Compare:"
    echo "  - outputs/original/submission.csv"
    echo "  - outputs/comparison/submission.csv"
    echo "  - data/test_annotations.csv"
else
    echo ""
    echo "ℹ️  No ground truth available for automatic scoring"
    echo "Visual inspection of outputs recommended"
fi

echo ""
echo "=========================================="
echo "✅ Test completed successfully!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Original: outputs/original/"
echo "  - Comparison: outputs/comparison/"
echo ""
