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
make notebook notebook_name=fingerprinting_ble_3_classes_1_server 2>&1 | tee /project/sdr4iot-ml/build/log_fingerprinting_ble_3_classes_1_server_$(date +%s).txt
make notebook notebook_name=fingerprinting_ble_3_classes_3_servers 2>&1 | tee /project/sdr4iot-ml/build/log_fingerprinting_ble_3_classes_3_servers_$(date +%s).txt

set +v

sleep 2

echo 'All done.'