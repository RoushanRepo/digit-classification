#!/bin/bash

# Build the Docker image
docker build -t train_svm_img .

# Run the Docker container
docker run -v m22aie243_volume:/app/model train_svm_img

