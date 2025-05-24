# System OCR

## Instrukcje postawienia lokalnej wersji
### 1. Załóż środowisko lokalne
Tutaj zastosowano condę, można też użyć venv:
```bash
conda create -n django_test python=3.13
conda activate django_test

cd ~/PycharmProjects
git clone <link_do_repo> <nazwa_folderu_docelowego>
cd <nazwa_folderu_docelowego>

pip install -r requirements.txt

# aby poznać swoją wersję CUDA, uruchom 'nvidia-smi'

# Torch Cuda <12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Cuda >=12.6, <12.8
pip3 install torch torchvision torchaudio
# Cuda >=12.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# PaddleOCR Cuda <12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
# Cuda >=12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

pip install paddleocr

# For ReactJS
conda install -c conda-forge nodejs=22.13
```
### 2. Utwórz plik .env i dodaj tam odpowiednie zmienne środowiskowe
```bash
cd ocr
nano .env
############## ZAWARTOŚĆ .env ##############
# SECRET_KEY=<klucz_prywatny>
# UPLOADED_FILES=<katalog_plikow_lokalnych>
# FIREBASE_KEY=<adres_lokalny_jsona_wygenerowanego_w_firebase>
# GOOGLE_OAUTH2_CLIENT_ID=<id_klienta_google_auth>
# GOOGLE_OAUTH2_CLIENT_SECRET=<klucz_prywatny_google_auth>
# GOOGLE_OAUTH2_REDIRECT_URI=<link_powrotny_google_auth>
############## ZAWARTOŚĆ .env ##############
```
Czyli przykładowo, jeśli mój klucz prywatny to django-insecure-c-!bac#($x2etc, a katalog w którym
chciałbym lokalnie przechowywać pliki wrzucone na stronę testową to /home/janek/PycharmProjects/ocr/media,
to mój plik .env będzie miał następującą treść:
```bash
SECRET_KEY=django-insecure-c-!bac#($x2etc
UPLOADED_FILES=/home/janek/PycharmProjects/ocr/media
FIREBASE_KEY=<adres_lokalny_jsona_wygenerowanego_w_firebase>
GOOGLE_OAUTH2_CLIENT_ID=<id_klienta_google_auth>
GOOGLE_OAUTH2_CLIENT_SECRET=<klucz_prywatny_google_auth>
GOOGLE_OAUTH2_REDIRECT_URI=<link_powrotny_google_auth>
```
Nowy SECRET_KEY można wygenerować za pomocą polecenia
```bash
django-admin shell
```
uruchomionego w środowisku, gdzie jest zainstalowane django (patrz oficjalny tutorial). Wpisujemy
```bash
from django.core.management.utils import get_random_secret_key  
get_random_secret_key()
```
w powłokę, którą przed chwilą uruchomiliśmy, aby uzyskać nowy klucz prywatny dla Django.

Gotowy plik .env oraz json z firebase znajduje się na #mati-keygen, ale swoje klucze można wygenerować.
Json z firebase znajduje się w:
Project settings -> Service accounts -> Generate new private key
Pod linkiem:
https://console.firebase.google.com/u/0/project/io2025-d859f/overview
Zmienne googlowe można znaleźć w:
Google cloud services -> Google auth platform (najlepiej wyszukać w wyszukiwarce na górze strony) -> Clients
Tam możemy wybrać klienta i dodawać oraz edytować rzeczy takie jak redirect uri
Zmienne te są pod linkiem:
https://console.cloud.google.com/welcome?project=io2025-d859f
### 3. (conda) Zmień interpreter swojego projektu w PyCharmie na django_test:
![Ustawienia interpretera](interpreter.png)

### 4. Dane do modelu
Na ten moment interesują nas pierwsze dwa datasety.
```bash
#IAM dataset do walidacji i testowania (oba linki wymagają logowania)
https://www.kaggle.com/datasets/ngkinwang/iam-dataset
https://fki.tic.heia-fr.ch/DBs/iamDB/data/lines.tgz
```
```bash
#Polish handwritten letters dataset do fine-tuning
https://www.kaggle.com/datasets/westedcrean/phcd-polish-handwritten-characters-database
```
Umiejscowić archiwa tak, żeby miały następującą strukturę
```bash
model/
├── modelbase.py
├── trocr.py
├── (...)
├── archive.zip #Kaggle
├── lines.tgz #FKI
└── setup_datasets.sh
```
A następnie uruchomić z poziomu katalogu model/ skrypt ./setup_datasets.sh.

### Instrukcje tworzenia od zera
```bash
conda create -n django_test python
conda activate django_test

pip install -r requirements.txt

django-admin startproject ocr
cd ocr

python manage.py startapp application

mkdir application/templates/application
(...)
```
