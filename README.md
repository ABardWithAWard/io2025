# System OCR



### Dane do logowania administratora

```bash

Email:    admin@example.com

Password: admin

```



## Instrukcje postawienia lokalnej wersji

### 1. Załóż środowisko lokalne

Tutaj zastosowano condę, można też użyć venv:

```bash

conda create -n django_test python=3.12.9

conda activate django_test



# For ReactJS

conda install -c conda-forge nodejs=22.13



cd ~/PycharmProjects

git clone <link_do_repo> <nazwa_folderu_docelowego>

cd <nazwa_folderu_docelowego>



# aby poznać swoją wersję CUDA, uruchom 'nvidia-smi'



# PaddleOCR Cuda <12.6

python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Cuda >=12.6

python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/



pip install -r requirements.txt



cd frontend

npm install

npm run build





cd ..

# przed tym krokiem wykonaj kroki nr 3 i 4

python manage.py collectstatic --noinput



# upewnij się, ze jesteś w root directory projektu

python ocr/manage.py runserver_plus --cert-file cert.pem --key-file key.pem 0.0.0.0:8000

```




```bash

# w przypadku problemów

conda deactivate django_test

conda remove -n django_test --all --keep-env

conda activate django_test



# przekiopiuj requirements_universal do requirements.txt



# PaddleOCR Cuda <12.6

python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Cuda >=12.6

python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/



pip install paddleocr



pip install -r requirements.txt

(...)

```



### 2. (conda) Zmień interpreter swojego projektu w PyCharmie na django_test:

![Ustawienia interpretera](interpreter.png)



### 3. Utwórz plik .env i dodaj tam odpowiednie zmienne środowiskowe

#### Zawartość obu plików .env znajduje się na kanale wewnętrznym. Dostosuj strukturę scieżek do systemu operacyjnego.

```bash

cd ocr

nano .env

# (...) uzupełnienie ocr/.env



cd ../frontend

nano .env

# (...) uzupełnienie frontend/.env

```



```bash

# nieuzupełniona zawartość ocr/.env

SECRET_KEY=

UPLOADED_FILES=

FIREBASE_KEY=

GOOGLE_OAUTH2_CLIENT_ID=

GOOGLE_OAUTH2_CLIENT_SECRET=

GOOGLE_OAUTH2_REDIRECT_URI=

```



```bash

# nieuzupełniona zawartość frontend/.env

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



Gotowy plik .env oraz json z firebase znajduje się na kanale wewnętrznym, ale swoje klucze można wygenerować.

Json z firebase znajduje się pod linkiem

```bash

https://console.firebase.google.com/u/0/project/io2025-d859f/overview

```

po wejściu w zakładkę:

```bash

Project settings -> Service accounts -> Generate new private key

```



Zmienne googlowe można znaleźć w:

```bash

Google Cloud Services -> Google Auth Platform (najlepiej wyszukać w wyszukiwarce na górze strony) -> Clients

```

Tam możemy wybrać klienta i dodawać oraz edytować rzeczy takie jak redirect url.



### 4. Key-gen

#### Windows

Po wszystkim nalezy takze wygenerowac certyfikat i go podpisac (mozna rowniez uzyc istniejacego, jesli ktos posiada)

```bash

./certgen.ps1

```

Nastepnie nalezy uzupelnic pola na terminalu

#### Linux

```bash

# uruchom w root directory projektu

openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

```



### 5. Dane do modelu

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