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

#### 2. Installing Dependencies

If you haven't installed the required dependencies, use the following commands to install them:

~~~shell
pip install transformers  

pip install torch  

pip install accelerate  

pip install librosa  

sudo apt install ffmpeg
~~~

---

#### 3. Optional: Creating a Virtual Environment

It's recommended to create a virtual environment to manage project dependencies. Follow these steps:

~~~shell
python -m venv .env

source .env/bin/activate
~~~

---

#### 4. Execution

To run the project, execute the following command:

~~~shell
python nlp1.py audio_file_path=/path/to/AUDIO_FILE.mp3
~~~

Replace **/path/to/AUDIO_FILE.mp3** with the path to your audio file. A sample audio file in **Hindi** language is present in the **codePython** folder.

--- 

#### 5. Output

~~~json
{
    "resultTranscription": {
      "text": " इसके बाद हम एक ऐसे खिलाड़ी के बारे में बात करेंगे जिससे इस वर्ल्ड कप में भारतिय फैंस को काफी उमीदे रहेंगे इस खिलाड़ी के बारे में मैं आपको बस इतना बता दूं कि ये वो खिलाड़ी हैं जिसने इंटरनेशनल क्रिकेट में अपनी एंट्री का एलान औस्ट्रेलिया के गेंज जोरदार प्रदर्शन से किया आज से वर्ल्ड कप के दारान हर रोज हम आपको मिलते रहेंगे EAM Cricket World Cup 2007 अपडेट में",
      "chunks": [
        {
          "timestamp": [
            0.0,
            5.42
          ],
          "text": " इसके बाद हम एक ऐसे खिलाड़ी के बारे में बात करेंगे जिससे इस वर्ल्ड कप में भारतिय फैंस को काफी उमीदे रहेंगे"
        },
        {
          "timestamp": [
            5.42,
            13.46
          ],
          "text": " इस खिलाड़ी के बारे में मैं आपको बस इतना बता दूं कि ये वो खिलाड़ी हैं जिसने इंटरनेशनल क्रिकेट में अपनी एंट्री का एलान औस्ट्रेलिया के गेंज जोरदार प्रदर्शन से किया"
        },
        {
          "timestamp": [
            13.62,
            19.52
          ],
          "text": " आज से वर्ल्ड कप के दारान हर रोज हम आपको मिलते रहेंगे EAM Cricket World Cup 2007 अपडेट में"
        }
      ]
    },
    "resultTranslation": {
      "text": " Now we will talk about a player who will be a big fan of Indian fans in this World Cup. I will tell you that he is a player who has made his entry in international cricket with a strong Australian against. From today during the World Cup, we will meet you every day in EAM Cricket World Cup 2007 Update.",
      "chunks": [
        {
          "timestamp": [
            0.0,
            5.6
          ],
          "text": " Now we will talk about a player who will be a big fan of Indian fans in this World Cup."
        },
        {
          "timestamp": [
            5.6,
            13.6
          ],
          "text": " I will tell you that he is a player who has made his entry in international cricket with a strong Australian against."
        },
        {
          "timestamp": [
            13.6,
            19.64
          ],
          "text": " From today during the World Cup, we will meet you every day in EAM Cricket World Cup 2007 Update."
        }
      ]
    },
    "resultClassifier": [
      {
        "label": "POSITIVE",
        "score": 0.999645471572876
      }
    ]
  }
~~~

---  

#### 6. API CALL

A java project for invoking the python script via an API call is available.

~~~shell
curl --location --request POST 'http://localhost:8080/upload' \
--form 'file=@"/home/anantha/ARC/AIML_NLP/codePython/HIN_M_AbhishekS.mp3"'
~~~