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
make tune nb_server=1 nn_type=ResNet exp_name=$(date +%s) max_trial=50 2>&1 | tee /project/sdr4iot-ml/build/log_tune_ResNet_$(date +%s).txt
make tune nb_server=1 nn_type=AlexNet exp_name=$(date +%s) max_trial=50 2>&1 | tee /project/sdr4iot-ml/build/log_tune_AlexNet_$(date +%s).txt
make tune nb_server=1 nn_type=ConvRNN exp_name=$(date +%s) max_trial=50 2>&1 | tee /project/sdr4iot-ml/build/log_tune_ConvRNN_$(date +%s).txt

set +v

sleep 2

echo 'All done.'