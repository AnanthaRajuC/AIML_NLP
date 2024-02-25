## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### 1. Checking Version Details

Before starting, ensure that you have the required applications installed. Run the following commands to check their versions:

~~~shell
python --version
python3 --version
~~~

~~~shell
pip --version
pip list 
~~~

Also, verify the version of PyTorch by running:

~~~shell
python -c "import torch; print(torch.__version__)"
pip3 show torch
~~~

---  

#### 2. Optional: Creating a Virtual Environment

It's recommended to create a virtual environment to manage project dependencies. Follow these steps:

~~~shell
python -m venv .env

source .env/bin/activate
~~~

---

#### 3. Installing Dependencies

If you haven't installed the required dependencies, use the following commands to install them:

~~~shell
pip install transformers  

pip install torch  

pip install accelerate  

pip install librosa  

sudo apt install ffmpeg
~~~

---

#### 4. Execution

To run the project, execute the following command:

~~~shell
python nlp1.py audio_file_path=/path/to/AUDIO_FILE.mp3
~~~

Replace /path/to/AUDIO_FILE.mp3 with the path to your audio file. A sample audio file in Hindi language is present in the codePython folder.