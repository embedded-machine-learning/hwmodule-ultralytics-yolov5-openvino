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
  MODELNAME=`echo $MYFILENAME | sed 's/convert_yolo_onnx_to_ir_//' | sed 's/.sh//'`
  echo Selected model: $MODELNAME
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

convert_to_ir()
{
  echo Apply to model $MODELNAME with precision $PRECISION
  echo "Define API config file"
  
  #APIFILEEFF=$SCRIPTPREFIX/hardwaremodules/openvino/openvino_conversion_config/efficient_det_support_api_v2.4.json
  #APIFILESSD=$SCRIPTPREFIX/hardwaremodules/openvino/openvino_conversion_config/ssd_support_api_v2.4.json
  #APIFILEEFF=$OPENVINOINSTALLDIR/deployment_tools/model_optimizer/extensions/front/tf/efficient_det_support_api_v2.4.json
  #APIFILESSD=$OPENVINOINSTALLDIR/deployment_tools/model_optimizer/extensions/front/tf/ssd_support_api_v2.4.json
  #APIFILERCNN=$OPENVINOINSTALLDIR/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support_api_v2.4.json
  #APIFILE=ERROR

  #if [[ $MODELNAME == *"ssd"* ]]; then 
  #  APIFILE=$APIFILESSD
  #elif [[ $MODELNAME == *"effi"* ]]; then 
  #  APIFILE=$APIFILEEFF
  #elif [[ $MODELNAME == *"rcnn"* ]]; then 
  #  APIFILE=$APIFILERCNN
  #else
  #  echo "Error. API filename not found. $APIFILE"
  #fi
  
  #echo "Use this API file: $APIFILE"
  echo Apply ONNX Simplifier
  python -m onnxsim exported-models/$MODELNAME/saved_model.onnx exported-models/$MODELNAME/saved_model_simple.onnx
  
  echo "Start conversion of model $MODELNAME"
  #python $OPENVINOINSTALLDIR/deployment_tools/model_optimizer/mo.py \
  #--input_model exported-models/$MODELNAME/saved_model_simple.onnx \
  #--output_dir exported-models-openvino/$MODELNAME\_OV$PRECISION \
  #--input images \
  #--scale 255 \
  #--data_type $PRECISION
  
  python $OPENVINOINSTALLDIR/deployment_tools/model_optimizer/mo.py \
  --input_model exported-models/$MODELNAME/saved_model_simple.onnx \
  --output_dir exported-models-openvino/$MODELNAME\_OV$PRECISION \
  --model_name $MODELNAME\_OV$PRECISION \
  -s 255 \
  --reverse_input_channels \
  --data_type $PRECISION \
  --output $OUTPUT_NODES
  
  #python $OPENVINOINSTALLDIR/deployment_tools/model_optimizer/mo_tf.py \
  #--data_type $PRECISION \
  #--input_model "exported-models/$MODELNAME/saved_model_simple.onnx" \
  #--output_dir "exported-models-openvino/$MODELNAME\_OV$PRECISION"
  
  #--saved_model_dir="exported-models/$MODELNAME/saved_model" \
  #--tensorflow_object_detection_api_pipeline_config=exported-models/$MODELNAME/pipeline.config \
  #--transformations_config=$APIFILE \
  #--reverse_input_channels \
  #--data_type $PRECISION \
  #--output_dir=exported-models-openvino/$MODELNAME\_OV$PRECISION
  echo "Conversion finished"
}

###
# Main body of script starts here
###

echo #==============================================#
echo # CDLEML Process TF2 Object Detection API
echo #==============================================#

# Constant Definition
USEREMAIL=alexander.wendt@tuwien.ac.at
#MODELNAME=tf2oda_efficientdet_512x384_pedestrian_D0_LR02
MODELNAME=tf2oda_ssdmobilenetv2_300x300_pets_D100
PYTHONENV=tf24
BASEPATH=`pwd`
SCRIPTPREFIX=../../scripts-and-guides/scripts
MODELSOURCE=jobs/*.config
HARDWARENAME=IntelNUC
LABELMAP=label_map.pbtxt

# IMPORTANT: The output nodes of YoloV5 have to be defined separately for each network. Look 
# at the network in netron and put the node numbers here. 
# The Conv numbers are the same for all resolutions of a certain model
OUTPUT_NODES=Conv_245,Conv_294,Conv_343

#Openvino installation directory for the model optimizer
#OPENVINOINSTALLDIR=/opt/intel/openvino_repo/openvino
OPENVINOINSTALLDIR=/opt/intel/openvino_2021.4.582
PRECISIONLIST="FP16 FP32"
#PRECISIONLIST="FP16"

#Extract model name from this filename
get_model_name

#Setup python environment
setup_env

#Setup openvino environment
source $OPENVINOINSTALLDIR/bin/setupvars.sh

#echo "Setup task spooler socket."
. ~/tf2odapi/init_eda_ts.sh

#Extract height and width from model
get_width_and_height

#echo "Start training of $MODELNAME on EDA02" | mail -s "Start training of $MODELNAME" $USEREMAIL

echo Apply to model $MODELNAME
get_width_and_height

#Get image resolution from model name

alias python=python3

echo #====================================#
echo # Convert TF2 Model to OpenVino Intermediate Representation
echo #====================================#

for PRECISION in $PRECISIONLIST
do
  #echo "$f"
  #MODELNAME=`basename ${f%%.*}`
  echo $PRECISION
  convert_to_ir
  
done
