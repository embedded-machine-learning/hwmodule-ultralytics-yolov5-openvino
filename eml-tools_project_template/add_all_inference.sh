#!/bin/bash

# put in execution folder

echo "Run complete execution process: execute converted model, execute TF2 model"

echo No conversion yet as it is necessary to first convert all models and then execute

#./add_folder_conv_ir.sh
./add_folder_infopenvino_jobs.sh
./add_folder_infpt_jobs.sh