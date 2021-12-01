#!/bin/bash

#BASESCRIPT=tf2oda_train_eval_export
#BASESCRIPT=tf2_inf_eval_saved_model

# Functions
add_job()
{
  echo "Generate Training Script for $MODELNAME"
  cp $SCRIPTBASENAME\_TEMPLATE.sh $SCRIPTBASENAME\_$MODELNAME.sh
  echo "Add task spooler jobs for $MODELNAME to the task spooler"
  echo "Shell script tf2_inf_eval_saved_model_$MODELNAME.sh"
  ts -L $USERNAME\_$MODELNAME $CURRENTFOLDER/$SCRIPTBASENAME\_$MODELNAME.sh
}

###
# Main body of script starts here
###

# Constant Definition
USERNAME=wendt
USEREMAIL=alexander.wendt@tuwien.ac.at
CURRENTFOLDER=`pwd`
MODELSOURCE=exported-models-openvino/*
SCRIPTBASENAME=openvino_inf_eval_yolo_onnx

echo "Setup task spooler socket."
. ./init_ts.sh


#Send start mail
ts -L Send_Start $CURRENTFOLDER/sendmail_Start_OpenVino_IntelNUC.sh

for f in $MODELSOURCE
do
  #echo "$f"
  MODELNAME=`basename ${f%%.*}`
  echo $MODELNAME
  add_job
  
  # take action on each file. $f store current file name
  #cat $f
done

#Send stop mail
ts -L Send_stop $CURRENTFOLDER/sendmail_Stop_OpenVino_IntelNUC.sh

