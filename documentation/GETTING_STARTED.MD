## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

~~~shell
python --version
python3 --version
~~~

~~~shell
pip --version
~~~

~~~shell
python -c "import torch; print(torch.__version__)"
pip3 show torch
~~~

~~~shell
pip3 list | grep tensorflow
pip list | grep tensorflow
~~~

~~~shell
pip list 
~~~

---  

~~~shell
python -m venv .env

source .env/bin/activate
~~~

pip install transformers
pip install torch
pip install datasets
pip install accelerate
pip install librosa
sudo apt install ffmpeg
