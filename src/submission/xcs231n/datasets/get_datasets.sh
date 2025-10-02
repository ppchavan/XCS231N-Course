#!/bin/bash

# Check if wget is available, otherwise use curl
if command -v wget >/dev/null 2>&1; then
  echo "Using wget to download datasets..."
  wget http://cs231n.stanford.edu/imagenet_val_25.npz
  if [ ! -d "coco_captioning" ]; then
      wget "http://cs231n.stanford.edu/coco_captioning.zip"
      unzip coco_captioning.zip
      rm coco_captioning.zip
  fi

elif command -v curl >/dev/null 2>&1; then
  echo "wget not found, using curl to download datasets..."
  curl -L http://cs231n.stanford.edu/imagenet_val_25.npz -o imagenet_val_25.npz
  if [ ! -d "coco_captioning" ]; then
      curl -L "http://cs231n.stanford.edu/coco_captioning.zip" -o coco_captioning.zip
      unzip coco_captioning.zip
      rm coco_captioning.zip
  fi

else
  echo "Error: Neither wget nor curl is available. Please install one of them."
  exit 1
fi