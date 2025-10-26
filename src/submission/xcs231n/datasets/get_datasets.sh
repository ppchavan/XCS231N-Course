if [ ! -d "cifar-10-batches-py" ]; then
  # Check if wget is available, otherwise use curl
  if command -v wget >/dev/null 2>&1; then
    echo "Using wget to download datasets..."
    wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    rm cifar-10-python.tar.gz
    wget http://cs231n.stanford.edu/imagenet_val_25.npz
  elif command -v curl >/dev/null 2>&1; then
    echo "wget not found, using curl to download datasets..."
    curl -L http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    rm cifar-10-python.tar.gz
    curl -L http://cs231n.stanford.edu/imagenet_val_25.npz -o imagenet_val_25.npz
  else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
  fi
fi