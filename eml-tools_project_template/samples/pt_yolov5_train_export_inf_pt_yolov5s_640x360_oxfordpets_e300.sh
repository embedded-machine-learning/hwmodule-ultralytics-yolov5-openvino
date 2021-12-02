#!/bin/bash

###
# Functions
###

setup_env()
{
  # Init environment
  . ./init_env.sh
  # Init task spooler
  . ./init_ts.sh
  
  alias python=python3.8
}

get_model_name()
{
  MYFILENAME=`basename "$0"`
  MODELNAME=`echo $MYFILENAME | sed 's/pt_yolov5_train_export_inf_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME
}

train_model()
{
  python train.py \
  --data $YOLODATA.yaml \
  --cfg $YOLOVERSION.yaml \
  --hyp data/hyps/hyp.scratch.yaml \
  --weights ./weights/$YOLOWEIGHTS.pt \
  --name $MODELNAME \
  --exist-ok \
  --batch-size $BATCHSIZE \
  --img $YOLOIMGSIZE \
  --epochs $EPOCHS
  
  #python /srv/cdl-eml/tf2odapi/yolov5/train.py \
  #--data ./peddet.yaml \
  #--cfg /srv/cdl-eml/tf2odapi/yolov5/models/yolov5s.yaml \
  #--weights '/srv/cdl-eml/tf2odapi/yolov5/weights/yolov5s.pt' \
  #--batch-size 64
  
  echo "#Copy trained model to model export folder"
  # Copy exported models to correct folders
  mkdir -p ./exported-models/$MODELNAME
  cp ./runs/train/$MODELNAME/weights/best.pt ./exported-models/$MODELNAME/
  mv ./exported-models/$MODELNAME/best.pt ./exported-models/$MODELNAME/saved_model.pt
}

export_model()
{
  python export.py \
  --weights ./exported-models/$MODELNAME/saved_model.pt \
  --simplify \
  --batch 1 \
  --opset 12 \
  --img $YOLOIMGSIZE
}

infer_with_model()
{
  python detect.py \
  --weights ./exported-models/$MODELNAME/saved_model.pt \
  --source $DATASET/images/val/ \
  --name $MODELNAME \
  --exist-ok \
  --conf-thres 0.5 \
  --save-txt \
  --save-conf \
  --nosave
  
  #--save-crop \
  
  #Move detections to results
  rm -r ./results/$MODELNAME/$HARDWARENAME/*
  mkdir -p ./results/$MODELNAME/$HARDWARENAME
  mv ./runs/detect/$MODELNAME/* ./results/$MODELNAME/$HARDWARENAME
  
}

measure_latency()
{
  echo "#====================================#"
  echo "# Perform Latency Evaluation"
  echo "#====================================#"
  
  python $SCRIPTPREFIX/inference_evaluation/pt_latency_from_saved_model.py \
  --image_dir=$DATASET/images/val \
  --latency_out=results/latency_$HARDWARENAME.csv \
  --model_name=$MODELNAME \
  --model_path=./exported-models/$MODELNAME/saved_model.pt

}

perform_validation()
{
  # Yolos built in evaluation function
  python val.py \
  --weights ./exported-models/$MODELNAME/saved_model.pt \
  --data oxford_pets_debug.yaml \
  --batch-size 1 \
  --img $YOLOIMGSIZE \
  --name $MODELNAME \
  --save-json \
  --save-txt \
  --iou-thres 0.5 \
  --save-conf \
  --conf-thres 0.5 \
  --exist-ok \
  --task val
  #2>&1 | tee SomeFile.txt

}

evaluate_model()
{
  echo "#====================================#"
  echo "# Convert Yolo Detections to Tensorflow Detections CSV Format"
  echo "#====================================#"
  echo "Convert Yolo tp TF CSV Format"
  python $SCRIPTPREFIX/conversion/convert_yolo_to_tfcsv.py \
  --annotation_dir="results/$MODELNAME/$HARDWARENAME/labels" \
  --image_dir="$DATASET/images/val" \
  --output="results/$MODELNAME/$HARDWARENAME/detections.csv"
  
  
  echo "#====================================#"
  echo "# Convert Detections to Pascal VOC Format"
  echo "#====================================#"
  echo "Convert TF CSV Format similar to voc to Pascal VOC XML"
  python $SCRIPTPREFIX/conversion/convert_tfcsv_to_voc.py \
  --annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
  --output_dir="results/$MODELNAME/$HARDWARENAME/det_xmls" \
  --labelmap_file="$DATASET/annotations/label_map.pbtxt"


  echo "#====================================#"
  echo "# Convert to Pycoco Tools JSON Format"
  echo "#====================================#"
  echo "Convert TF CSV to Pycoco Tools csv"
  python $SCRIPTPREFIX/conversion/convert_tfcsv_to_pycocodetections.py \
  --annotation_file="results/$MODELNAME/$HARDWARENAME/detections.csv" \
  --output_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json"

  echo "#====================================#"
  echo "# Evaluate with Coco Metrics"
  echo "#====================================#"
  echo "coco evaluation"
  python $SCRIPTPREFIX/inference_evaluation/eval_pycocotools.py \
  --groundtruth_file="$DATASET/annotations/coco_val_annotations.json" \
  --detection_file="results/$MODELNAME/$HARDWARENAME/coco_detections.json" \
  --output_file="results/performance_$HARDWARENAME.csv" \
  --model_name=$MODELNAME \
  --hardware_name=$HARDWARENAME \
  --index_save_file="./tmp/index.txt"

  echo "#====================================#"
  echo "# Merge results to one result table"
  echo "#====================================#"
  echo "merge latency and evaluation metrics"
  python $SCRIPTPREFIX/inference_evaluation/eval_merge_results.py \
  --latency_file="results/latency_$HARDWARENAME.csv" \
  --coco_eval_file="results/performance_$HARDWARENAME.csv" \
  --output_file="results/combined_results_$HARDWARENAME.csv"
}



###
# Main body of script starts here
###

echo "#==============================================#"
echo "# CDLEML Tool YoloV5 Training"
echo "#==============================================#"

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdetd0_320_240_coco17_pedestrian_all_LR002
#MODELNAME=pt_yolov5s_640x360_peddet_OLD
SCRIPTPREFIX=../../eml-tools
DATASET=../../../datasets/dataset-oxford-pets-val-debug
#DATASET=../../datasets/pedestrian_detection_graz_val_only_debug
HARDWARENAME=IntelNUC
# Set this variable true if the network shall be trained, else only inference shall be performed
TRAINNETWORK=false

############################################
### Yolo settings ##########################
############################################
# Yolo network configuration. ./models/[MODEL_NAME].yaml
#YOLOVERSION=yolov5s
#YOLOVERSION=yolov5m
#YOLOVERSION=yolov5l
#YOLOVERSION=yolov5x
YOLOVERSION=yolov5s_pedestrian

# Yolo pretrained weights. Location ./weights/[WEIGHT_NAME].pt.
# Weights for small versions of yolo, image size 640
YOLOWEIGHTS=yolov5s
#YOLOWEIGHTS=yolov5m
#YOLOWEIGHTS=yolov5l
#YOLOWEIGHTS=yolov5x
# Weights for big versions of yolo, image size 1280
#YOLOWEIGHTS=yolov5s6
#YOLOWEIGHTS=yolov5m6
#YOLOWEIGHTS=yolov5l6
#YOLOWEIGHTS=yolov5x6

# Yolo image size. Sizes: 640, 1280
YOLOIMGSIZE=640
#YOLOIMGSIZE=1280
#you only supply the longest dimension, --img 640. The rest is handled automatically.

# Yolo dataset reference. ./data/[DATASET].yaml
YOLODATA=peddet

# Set training batch size
BATCHSIZE=32
EPOCHS=300

############################################

#SCRIPTPREFIX=../../scripts-and-guides/scripts/training


#Extract model name from this filename
#MYFILENAME=`basename "$0"`
#MODELNAME=`echo $MYFILENAME | sed 's/tf2oda_train_eval_export_//' | sed 's/.sh//'`
#echo Selected model: $MODELNAME


# Environment preparation
setup_env

# Get model name
get_model_name

if [ "$TRAINNETWORK" = true ]
then
  #echo "$MODELNAME on EDA02 START training" | mail -s "$MODELNAME on EDA02 START training" $USEREMAIL
  
  echo "#====================================#"
  echo "#Train model"
  echo "#====================================#"
  echo model used $MODELNAME
  train_model

  echo "#====================================#"
  echo "#Export inference graph to ONNX"
  echo "#====================================#"
  echo "Export best weights for $MODELNAME to ONNX"
  export_model
  
  #echo "$MODELNAME on EDA02 COMPLETED training" | mail -s "$MODELNAME on EDA02 COMPLETED training" $USEREMAIL
  
else
  echo "No training will take place, only inference"
fi

echo "#====================================#"
echo "#Infer validation images"
echo "#====================================#"

#echo "$MODELNAME on EDA02 Start Inference" | mail -s "$MODELNAME on EDA02 Start Inference" $USEREMAIL

echo "Perform accuracy inference"
infer_with_model

echo "Perform latency measurement"
measure_latency

echo "Convert values and create evaluation"
evaluate_model


#echo "$MODELNAME on EDA02 COMPLETED inference" | mail -s "$MODELNAME on EDA02 COMPLETED inference" $USEREMAIL

echo "#======================================================#"
echo "# Training, evaluation and export of the model completed"
echo "#======================================================#"
