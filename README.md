###Low-Cost to Crack Python CAPTCHA Libraries

####Description
-----
The code layout contained within the root of this directory is separated into three distinct parts.
- **benchmarking** - This directory serves as the local and external benchmarking tests to be used to test our segmentation and recognition model.
- **generate_train** - This directory serves as a hub to generate data and train on the fly for our recognition model or to generate a CSV file containing all the data needed to train our segmentation model.
- **utils** - This directory serves as a repository to hold only tools to validate data or help construct any framework.

Overall, the integrity of this project was to illustrate how using Python CAPTCHA libraries to train a CNN model relatively quick and with conjunction of other modules, crack external CAPTCHA imagery.

####Environment
-----
- Operating System(s): Windows 10 x64, Ubuntu 16.04 amd64
- Video Card(s): Nvidia GTX 1070 8GB
- RAM: 16GB
- Anaconda Virtual Environment for Python codebase execution

####Dependencies
-----
- Python Version **3.6.8**
- **requirements.txt** satisfies package dependencies
- **Optional** - We highly recommend that [CUDA](https://developer.nvidia.com/cuda-zone) is installed on any compatible machine. To install CUDA, follow this [tutorial](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

###Building the Model
----
####Setup
Begin the setup process by opening Anaconda Virtual Environment, if your on a Linux system then odds are your Anaconda installation has been exported to your current bash session. If not, then a launcher should be readily available (This is the case on Windows machines).

***Environment Setup***

`$ conda create -n ENVIRONMENT_NAME python=3.6 pip` 

***Activating the Environment***

`$ conda activate ENVIRONMENT_NAME`

***Installing requirements.txt***

`$ pip install -r requirements.txt`

***Setup your paths***
Upon every activation of your Anaconda Virtual Environment, the respective run file(s) must be ran in order for Python to properly communicate with the various directories contained within the project.

**Windows**

`> run.bat`

**Linux**

`$ source run.src`

####Generating & Training the Model
Navigate to the **generate_train/** directory within the root of the repository, here there are two files to assist with either building the segmentation framework data or to train our recognition model.

***Data generation for the segmentation model***
Our segmentation model comes from [TensorFlows Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) repository on the TensorFlow GitHub repository. If you want to build your own segmentation model then I suggest following [this](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) tutorial as it has outlined key points and has covered a great deal of detail into generating a model. Our base model for the segmentation model is the **atrous** model found within TensorFlows [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) repository on GitHub.

To generate the data, invoke:

`$ python od_generate.py -s NUMBER_OF_SAMPLES -l MAX LENGTH OF CAPTCHA -f fonts/ -d NEW DIRECTORY -t NUMBER OF THREADS`

***Generate and train the recognition model***
This part of the process doesnt require external CSV files being generated, rather its done all in memory. **Batch size (-b)** controls the amount of files generated per batch to be sent to the model to be trained upon. **Generations (-g)** controls the amount of generations done until model completion and can be stopped at any time as the model is saved per generation.

*Sample*

`$ python generate_train.py -b 8192 -g 5 -f fonts/`

###Benchmarking our model
----
After generating our recognition model, we must first install the segmentation model framework into our benchmarking directory. This file may be found on our [Releases Page](https://github.com/IAmAbszol/CAPTCHACrackingTools/releases) on this GitHub repository. Instructions are provided but in short, make sure the **inference_graph/** directory is placed within the root of the **benchmarking/** directory.

Place the generated **models/** directory inside the root of the benchmarking directory.

Finally we have two files inside this directory to assist in benchmarking your newly created model.

####Benchmarking locally
This benchmark focuses on the CAPTCHA we created and testing our TOD framework against our Peak Segmentation framework.

`$ python local_benchmark.py`

The resulting output provides detailed information and imagery to understand where our model faults.

####Benchmarking externally
Before March, a Python script was setup to rip CAPTCHAs off a specified website. They have since updated their site to reCAPTCHA v3 whose CAPTCHA may not be ripped. The resulting directory is of numerous samples taken from the site.

`$ python external_benchmark.py`


###Other
----

####Changing characters
The **utils/** directory located in the root of the repository contains a script *helper.py* which assists in certain training aspects of the model. Within this script there contains a set of characters to be used in the training and evaluation of our model, keeps this constant during the whole training and evaluation process.

`training_characters = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '@', '#', '$','%', '^', '&', '*']
`

####Peak segmentation framework
The framework was the first attempt at segmenting the characters and has since been deprecated and serves as a baseline for comparison of the success of the TOD segmentation framework.