# OCR Note-scanning Site

## Local Version Setup Instructions
### 1. Set up local environment
To check your CUDA version, run:
```bash
nvidia-smi
```
on both Windows and Linux systems. Depending on the version,
select the appropriate OCR module version according to the instructions in the rest of the README.

Conda is used here, but you can also use venv:
```bash
cd ~/PycharmProjects
git clone <repo_link> <target_folder_name>
cd <target_folder_name>

conda create -n django_test python=3.12.9
conda activate django_test
```

# For React
```bash
conda install -c conda-forge nodejs=22.13
```

# PaddleOCR Cuda <12.6
```bash
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```
# Cuda >=12.6
```bash
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
```bash
pip install -r requirements.txt

cd frontend && npm install && npm run build

cd ..
```

Create .env files and add the appropriate environment variables. The contents of both .env files
are available on the internal channel. Adjust path structures to your operating system.

```bash
cd ocr
nano .env
# (...) fill in ocr/.env
Unfilled contents of ocr/.env:
bashSECRET_KEY=
UPLOADED_FILES=
FIREBASE_KEY=
GOOGLE_OAUTH2_CLIENT_ID=
GOOGLE_OAUTH2_CLIENT_SECRET=
GOOGLE_OAUTH2_REDIRECT_URI=
And the second .env:
bashcd ../frontend
nano .env
# (...) fill in frontend/.env
```

Unfilled contents of frontend/.env:
```bash
REACT_APP_GOOGLE_OAUTH2_CLIENT_ID=
REACT_APP_FIREBASE_KEY=
REACT_APP_TYPE=
REACT_APP_PROJECT_ID=
REACT_APP_PRIVATE_KEY_ID=
REACT_APP_PRIVATE_KEY=
REACT_APP_CLIENT_EMAIL=
REACT_APP_CLIENT_ID=
REACT_APP_AUTH_URI=
REACT_APP_TOKEN_URI=
REACT_APP_AUTH_PROVIDER_X509_CERT_URL=
REACT_APP_CLIENT_X509_CERT_URL=
REACT_APP_UNIVERSE_DOMAIN=
```

Now you need to generate and set up certificates.
After running the commands, fill in the answers to questions in the terminal (location, email, etc.)
Windows
```bash
cd ..
./certgen.ps1
```

Linux
```bash
cd ..
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

Final steps to run the application:
```bash
python ocr/manage.py collectstatic --noinput
python ocr/manage.py migrate
```

make sure you're in the project root directory
```bash
python ocr/manage.py runserver_plus --cert-file cert.pem --key-file key.pem 0.0.0.0:8000
```
2. (dev) Change your project interpreter in PyCharm to django_test:
![Interpreter settings](interpreter.png)
3. (dev) Generating and setting new keys
A new SECRET_KEY can be generated using the command:
```bash
django-admin shell
```
run in an environment where Django is installed (see official tutorial). Enter:

```python
from django.core.management.utils import get_random_secret_key  
get_random_secret_key()
```
in the shell you just launched to get a new private key for Django.
A ready .env file and Firebase json are available on the internal channel, but you can generate your own keys.
Firebase json can be found at:
```bash
https://console.firebase.google.com/u/0/project/io2025-d859f/overview
```
by going to the tab:
```bash
Project settings -> Service accounts -> Generate new private key`
```

Google variables can be found in:
```bash
Google Cloud Services -> Google Auth Platform (best to search in the search bar at the top) -> Clients
```
There we can select a client and add/edit things like redirect url.

4. (dev) Test data for the model
Currently, we're interested in the first two datasets.
```bash
#IAM dataset for validation and testing (both links require login)
https://www.kaggle.com/datasets/ngkinwang/iam-dataset
https://fki.tic.heia-fr.ch/DBs/iamDB/data/lines.tgz
```
For fine-tuning:
```bash
#Polish handwritten letters dataset for fine-tuning
https://www.kaggle.com/datasets/westedcrean/phcd-polish-handwritten-characters-database
```
Place the archives to have the following structure:
```bash
model/
├── modelbase.py
├── trocr.py
├── (...)
├── archive.zip #Kaggle
├── lines.tgz #FKI
└── setup_datasets.sh
```
Then run the ./setup_datasets.sh script from the model/ directory.

5. Running tests
Automatically:
From the application root folder, run run_tests.ps1

##### Manually:
First terminal:
```bash
cd frontend
npm start
```
Second terminal:
```bash
cd ocr
coverage erase
coverage run --rcfile=.coveragerc manage.py test
coverage html
start htmlcov/index.html
```
