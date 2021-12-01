# Running YoloV5 in OpenVino with EML Tools
In this folder, there is a template project for inference of trained, exported models of YoloV5 on OpenVino. In the following
procedure, instructions are provided to setup and run one or more networks and to extract the evaluations of the executions. All
evaluations are compatible with the EML tools.

## Setup

### Prerequisites
1. Setup the task spooler on the target device. Instructions can be found here: https://github.com/embedded-machine-learning/scripts-and-guides/blob/main/guides/task_spooler_manual.md

### Dataset
For validating the tool chain, download the small validation set from kaggle: https://www.kaggle.com/alexanderwendt/oxford-pets-cleaned-for-eml-tools

It contains a snall set that is used for inference validation in the structure that is compatible to the EML Tools. Put it in the following folder structure

To be able to use the dataset, it is necessary to have a trained model of yolov5 on the Oxford Pets dataset. It can be done here: TBD Link YoloV5 on Server.

### Generate EML Tools directory structure
The following steps are only necessary if you setup the EML tools for the first time on a device.
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
pip install lxml xmltodict tdqm beautifulsoup4

# Install OpenVino libraries
pip install onnx-simplifier networkx defusedxml

```

3. Copy the scripts from this folder to your project folder and execute ```chmod 777 *.sh``` in the yolov5 folder.

4. Execute ```setup_dirs.sh``` to create all necessary sub folders


### Modification of script files
The next step is to adapt the script files to the current environment.

#### Conversion script for ONNX models of yolov5 to OpenVino IR model
convert_yolo_onnx_to_ir_pt_yolov5l_640x360_peddet



## Running the system