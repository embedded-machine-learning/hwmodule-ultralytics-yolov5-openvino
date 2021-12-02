# Running YoloV5 in OpenVino with EML Tools
In this folder, there is a template project for inference of trained, exported models of YoloV5 on OpenVino. In the following
procedure, instructions are provided to setup and run one or more networks and to extract the evaluations of the executions. All
evaluations are compatible with the EML tools.

## Setup

### Prerequisites
Setup the task spooler on the target device. Instructions can be found here: https://github.com/embedded-machine-learning/scripts-and-guides/blob/main/guides/task_spooler_manual.md

### Dataset
For validating the tool chain, download the small validation set from kaggle: https://www.kaggle.com/alexanderwendt/oxford-pets-cleaned-for-eml-tools

It contains a small set that is used for inference validation in the structure that is compatible to the EML Tools. Put it in the following folder structure

To be able to use the dataset, it is necessary to have a trained model of yolov5 on the Oxford Pets dataset. It can be done here: https://github.com/embedded-machine-learning/hwmodule-ultralytics-yolov5-server.

### Generate EML Tools directory structure
The following steps are only necessary if you setup the EML tools for the first time on a device:

1. Create a folder for your datasets. Usually, multiple users use one folder for all datasets to be able to share them. Later on, in the 
training and inference scripts, you will need the path to the dataset.

2. Create the EML tools folder structure. The structure can be found here: https://github.com/embedded-machine-learning/eml-tools#interface-folder-structure

In your root directory, create the structure. Sample code
```
mkdir eml_projects
mkdir venv

```

3. Clone the EML tools repository into your workspace

```
git clone https://github.com/embedded-machine-learning/eml-tools.git
```

4. Create the task spooler script to be able to use the correct task spooler on the device. In our case, just copy

```./init_ts.sh``` to the workspace root. The idea is that all projects will use this task spooler.

### Project setup
1. Go to your project folder and clone the YoloV5 repository. Then rename it for your project.
```
cd eml_projects/
git clone https://github.com/ultralytics/yolov5.git
mv yolov5 yolov5-oxford-pets
```

2. Create a virtual environment for yolov5 in your venv folder. The venv folder is put outside of the project folder to 
avoid copying lots of small files when you copy the project folder. Conda would also be a good alternative.

```
# From root
cd ./venv
virtualenv -p python3.8 yolov5_pv38
source yolov5_pv38/bin/activate

# Install necessary libraries
python -m pip install --upgrade pip
pip install --upgrade setuptools cython wheel

# Install yolov5 libraries
cd ../eml_projects/yolov5-oxford-pets/
pip install -r requirements.txt

# Install EML libraries
pip install lxml xmltodict tdqm beautifulsoup4 pycocotools

# Install OpenVino libraries
pip install onnx-simplifier networkx defusedxml

```

3. Create a virtual environment for OpenVino 2021.4 in Python 3.6 as it does not cope with Python 3.8
4. 
```
# From root
cd ./venv
virtualenv -p python3.6.9 openvino_py36
source openvino_py36/bin/activate

# Install necessary libraries
python -m pip install --upgrade pip
pip install --upgrade setuptools cython wheel

# Install EML libraries
pip install lxml xmltodict tdqm beautifulsoup4 pycocotools pandas absl-py

# Install OpenVino libraries
pip install onnx-simplifier networkx defusedxml progress

```

4. Copy the scripts from this folder to your project folder and execute ```chmod 777 *.sh``` in the yolov5 folder.

5. Execute ```setup_dirs.sh``` to create all necessary sub folders

6. Copy all exported YoloV5 models to ```./exported-models```. The folder names look like this ```pt_yolov5s_640x360_oxfordpets_e300``` according to the interface from 
https://github.com/embedded-machine-learning/eml-tools#interface-network-folder-and-file-names. 
By default, each model folder contains a ```saved_model.pt```, a ```saved_model.torchscript.pt``` and a ```saved_model.onnx```. For our execution, 
the ```saved_model.onnx``` is necessary.

### Adaption of Script Files
The next step is to adapt the script files to the current environment.

#### Adapt Task Spooler Script
In ```init.ts.sh```, either adapt

```
export TS_SOCKET="/srv/ts_socket/CPU.socket"
chmod 777 /srv/ts_socket/CPU.socket
``` 

to your task spooler path or call another task spooler script in your EML Tools root.
```
. ../../init_eda_ts.sh
```

In ```init_env.sh```, adapt the ```source ../../venv/yolov5_pv38/bin/activate``` to your venv folder or conda implementation.

#### Conversion Script for ONNX models of Yolov5 to OpenVino IR Model
The first script to adapt is ```convert_yolo_onnx_to_ir_TEMPLATE.sh```. 

1. Get the output nodes of the ONNX file from https://github.com/embedded-machine-learning/hwmodule-ultralytics-yolov5-openvino#onnx-to-openvino

2. Copy and rename ```convert_yolo_onnx_to_ir_TEMPLATE.sh``` to ```convert_yolo_onnx_to_ir_[MODELNAME].sh```, e.g. 
```convert_yolo_onnx_to_ir_pt_yolov5s_640x360_oxfordpets_e300.sh```, where the MODELNAME is a exact match of the folder name of the model in ```exported-models```.
Now, this script will only be used to convert this model. Note that the MODELNAME will be extracted from the file name and information about the implementation will be extracted for the evaluation.

3. Edit the script and adapt the following constants for your conversion:

```
USEREMAIL=alexander.wendt@tuwien.ac.at			 #Your email
SCRIPTPREFIX=../../eml-tools    #There should be no need to change this
HARDWARENAME=IntelNUC							#Hardware identifier

# IMPORTANT: The output nodes of YoloV5 have to be defined separately for each network. Look 
# at the network in [Netron](https://netron.app/) and put the node numbers here. 
# The Conv numbers are the same for all resolutions of a certain model
OUTPUT_NODES=Conv_245,Conv_294,Conv_343

#Openvino installation directory for the model optimizer
OPENVINOINSTALLDIR=/opt/intel/openvino_2021.4.582
PRECISIONLIST="FP16 FP32"
```

#### Inference Execution Script for YoloV5 in Pytorch
The script ```pt_yolov5_train_export_inf_TEMPLATE.sh``` executes the model in the TEMPLATE on pytorch.

Adapt the following constants for your environment:
```
USERNAME=wendt   							# Name
USEREMAIL=alexander.wendt@tuwien.ac.at		# Email
SCRIPTPREFIX=../../eml-tools	# Should be no need to change this, if the EML Tools has been setup correctly
DATASET=../../../datasets/dataset-oxford-pets-val-debug  # Set the absolute or relative path of your EML Tools compatible dataset
HARDWARENAME=IntelNUC	# Set the hardware name

# Set this variable true if the network shall be trained, else only inference shall be performed
TRAINNETWORK=false
```

#### Inference Execution Script for YoloV5 in OpenVino
The script ```openvino_inf_eval_yolo_onnx_TEMPLATE.sh``` executes the TEMPLATE network on OpenVino.

Adapt the following constants for your environment:
```
SCRIPTPREFIX=../../eml-tools
HARDWARENAME=IntelNUC
DATASET=../../../datasets/dataset-oxford-pets-val-debug

# Openvino installation directory for the inferrer (not necessary the same as the model optimizer)
OPENVINOINSTALLDIR=/opt/intel/openvino_2021.4.582
HARDWARETYPELIST="CPU GPU MYRIAD"   # Set which devices you want to execute on. MYRIAD is the NCS2

```

#### Add Folder Jobs for Pytorch

```add_folder_infpt_jobs.sh``` reads all names from the exported-folder, copies the ```pt_yolov5_train_export_inf_TEMPLATE.sh```, replaces TEMPLATE with the model name and puts the created script into the task spooler. The task spooler then executes all models in the queue.

Adapt the following constants for your environment:
```
USERNAME=wendt     #Your name
USEREMAIL=alexander.wendt@tuwien.ac.at # Your Email
MODELSOURCE=exported-models/*    # No need to change this unless you are debugging.
```

#### Add Folder Jobs for OpenVino

```add_folder_infopenvino_jobs.sh``` reads all names from the exported-folder, copies the ```openvino_inf_eval_yolo_onnx_TEMPLATE.sh```, replaces TEMPLATE with the model name
and puts the created script into the task spooler. The task spooler then executes all models in the queue.

Adapt the following constants for your environment:
```
USERNAME=wendt	#Your name
USEREMAIL=alexander.wendt@tuwien.ac.at	#Your email
MODELSOURCE=exported-models-openvino/*	#No need to change this
```

## Running the system

1. Convert each exported model by running ```./convert_yolo_onnx_to_ir_[MODELNAME].sh```. You now get an FP16 and an FP32 model in ```./exported-models-openvino```
2. Run either ```add_folder_infopenvino_jobs.sh``` and ```add_folder_infpt_jobs.sh``` or just ```add_all_inference.sh```. If everything works, all inferences are executed
and the results are put in the ```./results``` folder under the name ```combined_results_[HARDWARENAME].csv```

## Embedded Machine Learning Laboratory

This repository is part of the Embedded Machine Learning Laboratory at the TU Wien. For more useful guides and various scripts for many different platforms visit 
our **EML-Tools**: **https://github.com/embedded-machine-learning/eml-tools**.

Our newest projects can be viewed on our **webpage**: **https://eml.ict.tuwien.ac.at/**
