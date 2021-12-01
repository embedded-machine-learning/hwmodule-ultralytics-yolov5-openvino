#!/bin/bash

###
# Functions
###

setup_env()
{
  # Environment preparation
  echo Activate environment $PYTHONENV
  #call conda activate %PYTHONENV%
  #Environment is put directly in the nuc home folder
  . ~/tf2odapi/init_eda_env.sh
}

get_model_name()
{
  MYFILENAME=`basename "$0"`
  MODELNAME=`echo $MYFILENAME | sed 's/openvino_inf_eval_yolo_onnx_//' | sed 's/.sh//'`
  echo Selected model based on folder name: $MODELNAME
}

get_width_and_height()
{
  elements=(${MODELNAME//_/ })
  #$(echo $MODELNAME | tr "_" "\n")
  #echo $elements
  resolution=${elements[2]}
  res_split=(${resolution//x/ })
  height=${res_split[0]}
  width=${res_split[1]}

  echo batch processing height=$height and width=$width

}

infer()
{
  echo Apply to model $MODELNAME with type $HARDWARETYPE

  echo '===================================='
  echo ' Infer with OpenVino'
  echo '===================================='
  echo "Start latency inference"
  python $SCRIPTPREFIX/hardwaremodules/intel/run_pb_bench_sizes.py \
  -openvino_path $OPENVINOINSTALLDIR \
  -hw $HARDWARETYPE \
  -batch_size 1 \
  -api $APIMODE \
  -niter 1000 \
  -nireq 1 \
  -xml ./exported-models-openvino/$MODELNAME/$MODELNAME.xml \
  -output_dir="results/$MODELNAME/$HARDWARENAME/openvino"

  #python $OPENVINOINSTALLDIR/deployment_tools/tools/benchmark_tool/benchmark_app.py \
  #--path_to_model "exported-models-openvino/$MODELNAME/saved_model_simple.xml" \
  #-niter 10 \
  #-nireq 1 \
  #-d $HARDWARETYPE

  echo '===================================='
  echo ' Convert Latencies'
  echo '===================================='
  echo "Add measured latencies to result table"
  python3 $SCRIPTPREFIX/hardwaremodules/intel/openvino_latency_parser.py \
  --avg_rep results/$MODELNAME/$HARDWARENAME/openvino/benchmark_average_counters_report_$HARDWARETYPE\_$APIMODE.csv \
  --inf_rep results/$MODELNAME/$HARDWARENAME/openvino/benchmark_report_$HARDWARETYPE\_$APIMODE.csv \
  --output_path results/latency_$HARDWARENAME.csv \
  --hardware_name $HARDWARENAME \
  --index_save_file="./tmp/index.txt"
  #::--save_new #False: Always append

  echo '===================================='
  echo ' Infer with OpenVino'
  echo '===================================='
  echo "Start accuracy/performance inference" 
  python $SCRIPTPREFIX/hardwaremodules/intel/test_write_results_yolov_3and5.py \
  -i $DATASET/images/val \
  -m ./exported-models-openvino/$MODELNAME/$MODELNAME.xml \
  -d $HARDWARETYPE \
  --detections_out results/$MODELNAME/$HARDWARENAME\_$HARDWARETYPE/detections.csv \
  --input_source onnx \
  --prob_threshold 0.5 \
  --iou_threshold 0.4 \
  --no-show \
  --labels $DATASET/annotations/labels.txt

  echo '===================================='
  echo ' Convert to Pycoco Tools JSON Format'
  echo '===================================='
  echo "Convert TF CSV to Pycoco Tools csv"
  python $SCRIPTPREFIX/conversion/convert_tfcsv_to_pycocodetections.py \
  --annotation_file=results/$MODELNAME/$HARDWARENAME\_$HARDWARETYPE/detections.csv \
  --output_file=results/$MODELNAME/$HARDWARENAME\_$HARDWARETYPE/coco_detections.json

  echo '===================================='
  echo ' Evaluate with Coco Metrics'
  echo '===================================='

  python $SCRIPTPREFIX/inference_evaluation/objdet_pycoco_evaluation.py \
  --groundtruth_file=$DATASET/annotations/coco_val_annotations.json \
  --detection_file=results/$MODELNAME/$HARDWARENAME\_$HARDWARETYPE/coco_detections.json \
  --output_file=results/performance_$HARDWARENAME.csv \
  --model_name=$MODELNAME \
  --hardware_name=$HARDWARENAME\_$HARDWARETYPE \
  --index_save_file=./tmp/index.txt
  
  echo '===================================='
  echo '  Merge results to one result table'
  echo '===================================='
  echo merge latency and evaluation metrics
  python3 $SCRIPTPREFIX/inference_evaluation/merge_results.py \
  --latency_file=results/latency_$HARDWARENAME.csv \
  --coco_eval_file=results/performance_$HARDWARENAME.csv \
  --output_file=results/combined_results_$HARDWARENAME.csv

}


###
# Main body of script starts here
###

echo #==============================================#
echo # CDLEML Process TF2 Object Detection API for OpenVino
echo #==============================================#

# Constant Definition
#USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdet_512x384_pedestrian_D0_LR02
#MODELNAME=tf2oda_ssdmobilenetv2_300x300_pets_D100_OVFP16
PYTHONENV=tf24
SCRIPTPREFIX=../../scripts-and-guides/scripts
HARDWARENAME=IntelNUC
DATASET=../../datasets/pedestrian_detection_graz_val_only_ss10
#DATASET=../../datasets/pedestrian_detection_graz_val_only_debug
#LABELMAP=label_map.pbtxt

#Openvino installation directory for the inferrer (not necessary the same as the model optimizer)
#OPENVINOINSTALLDIR=/opt/intel/openvino_2021
OPENVINOINSTALLDIR=/opt/intel/openvino_2021.4.582
#OPENVINOINSTALLDIR=/opt/intel/openvino_2021.3.394
APIMODE=sync
HARDWARETYPELIST="CPU GPU MYRIAD"
#HARDWARETYPELIST="CPU"

echo Extract model name from this filename
get_model_name

echo Extract height and width from model
get_width_and_height

echo Setup environment
setup_env

#echo "Start training of $MODELNAME on EDA02" | mail -s "Start training of $MODELNAME" $USEREMAIL

echo "Setup task spooler socket."
. ~/tf2odapi/init_eda_ts.sh

#Setup openvino environment
echo "Setup Openvino environment and variables"
source $OPENVINOINSTALLDIR/bin/setupvars.sh

alias python=python3

for HARDWARETYPE in $HARDWARETYPELIST
do
  #echo "$f"
  #MODELNAME=`basename ${f%%.*}`
  echo $HARDWARETYPE
  infer
  
done

