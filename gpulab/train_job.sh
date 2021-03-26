#!/bin/bash -e

echo 'This script shows the content of your project share, and some other info.'
echo 'Then it train SDR4IoT tensorflow model.'
echo "Date: $(date)"
echo "whoami: $(whoami)"
echo "Working directory: $(pwd)"
echo "GPULab Project: ${GPULAB_PROJECT}"
#echo "Size of /project: $(du -hs /project)"

set -v
ls /project
cd /project/sdr4iot-ml
make train nb_server=1 nn_type=AlexNet exp_name=$(date +%s) 2>&1 | tee /project/sdr4iot-ml/build/log_train_$(date +%s).txt
make train nb_server=1 nn_type=ResNet exp_name=$(date +%s) 2>&1 | tee /project/sdr4iot-ml/build/log_train_$(date +%s).txt
make train nb_server=1 nn_type=ConvRNN exp_name=$(date +%s) 2>&1 | tee /project/sdr4iot-ml/build/log_train_$(date +%s).txt

set +v

sleep 2

echo 'All done.'