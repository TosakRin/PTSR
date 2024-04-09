#!/bin/bash

# Check if two arguments were provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <size-in-kilobytes>"
    exit 1
fi

# Assign arguments to variables
DIR_PATH=$1
SIZE_LIMIT=$2

# Check if the directory exists
if [ ! -d "$DIR_PATH" ]; then
    echo "Error: Directory $DIR_PATH does not exist."
    exit 1
fi

# Find and delete files smaller than the specified size
find "$DIR_PATH" -type f -size -"${SIZE_LIMIT}"k -exec rm -v {} \;

echo "Deletion complete."

# clean empty directories
find "$DIR_PATH" -type d -empty -delete
