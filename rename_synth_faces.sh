#!/bin/bash

# Script to randomly order files in synth_faces/images and prepend sequential numbers

IMAGE_DIR="/home/iyu/ml-flextok/data/synth_faces/images"

# Check if directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory $IMAGE_DIR does not exist"
    exit 1
fi

# Create a temporary directory for processing
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Change to the image directory
cd "$IMAGE_DIR" || exit 1

# Get list of all files, shuffle them, and number them
echo "Generating random order and renaming files..."

# Count files first
file_count=$(find . -maxdepth 1 -name "*.png" -type f | wc -l)

if [ "$file_count" -eq 0 ]; then
    echo "No PNG files found in $IMAGE_DIR"
    exit 1
fi

echo "Found $file_count files"

# Use find to list files, shuffle them, number them, and rename
# This avoids the "Argument list too long" error
counter=1
find . -maxdepth 1 -name "*.png" -type f -printf '%f\n' | shuf | while read -r filename; do
    # Create zero-padded number with 6 digits
    new_name=$(printf "%06d_%s" "$counter" "$filename")

    # Move to temp directory with new name
    mv "$filename" "$TEMP_DIR/$new_name"

    counter=$((counter + 1))

    # Print progress every 1000 files
    if [ $((counter % 1000)) -eq 0 ]; then
        echo "Processed $counter files..."
    fi
done

echo "Moving renamed files back..."
# Move all files back from temp directory using find to avoid argument list limits
find "$TEMP_DIR" -maxdepth 1 -type f -exec mv {} . \;

# Clean up
rmdir "$TEMP_DIR"

echo "Done! Files have been randomly ordered and renamed."
echo "Example: $(ls | head -1)"
