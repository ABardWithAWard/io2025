# io2025

## Instrukcje konfiguracji lekcji

### Prerekwizyty
(OS: Linux)\
PyCharm, conda (Miniconda), git, gcc i g++ (inaczej paczki wheel w Pythonie nie będą się kompilować)

### Pierwsze kroki
Upewnij się, że masz wygenerowaną parę kluczy ssh. Jeśli tak nie jest, to uruchom ssh-keygen w terminalu. Interesuje
cię klucz z rozszerzeniem .pub.\
Dodaj swój klucz publiczny ssh do autoryzowanych na GitHubie (github.com/settings/profile, a następnie SSH and GPG keys).
Oszczędzisz sobie tym sposobem czas na autoryzacje przy korzystaniu z gita za pomocą HTTPS. Jeśli dodatkowo chcesz,
żeby twoje commity z PyCharma wyświetlały się później na GitHubie jako zielone ("Verified"), co będzie widać po wejściu
na twój profil, to zastosuj następujące zmiany (są globalne, ale zapewne można dostosować jedynie do tego projektu):
```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
```
Utwórz nowy branch na GitHubie o wybranej przez siebie nazwie _BRANCH_NAME_\
W terminalu, po wcześniejszym prawidłowym skonfigurowaniu condy (może być potrzebny restart), utwórz nowe środowisko
zgodne z wersją Pythona którą zamierzasz używać. Przykładowo, jeśli ma to być Python 3.12, to zastosuj:
```bash
conda create -n <nazwa_srodowiska> python=3.12 
```
#### Uwaga
Jeśli wcześniejsza komenda wyświetla coś w stylu 
```bash
CondaError: Run 'conda init' before 'conda activate'
```
To najprawdopodobniej źle postawiłeś condę :)\
Wejdź teraz w katalog na swoim komputerze, gdzie chciałbyś żeby znajdował się katalog projektowy (może to być np. ~/PycharmProjects).\
Wykonaj clone swojego brancha.
```bash
git clone git@github.com:ABardWithAWard/io2025.git -b <BRANCH_NAME> [nazwa_katalogu_do_którego_się_wykona_clone]
```
Uruchom PyCharma.\
W PyCharmie wybierz "New Project" wraz parametrami:\
Custom environment\
Select existing\
Type: Conda\
(...)\
W drop-down menu "environment" wybierz środowisko zgodnie z wybraną przez ciebie <nazwą_środowiska>.\
Powinieneś teraz mieć projekt w Pycharmie, który jest podpięty pod version control na repo projektu. Ponadto, interpreter
używany przez projekt to ten, który jest w utworzonym przez ciebie środowisku condy. Instalacja wszelkich package'ów
odbywa się więc za pomocą terminala (lub odpowiednich skryptów, które możesz utworzyć w katalogu projektu).\
Przykładowo wykonanie:
```bash
conda activate <nazwa_srodowiska>
pip install -r requirements.txt
```
zainstaluje dla środowiska condy <nazwa_srodowiska> paczki Pythonowe zawarte w pliku requirements.txt za pomocą pip.

### Wypakowanie plików z konkretnej lekcji do głównego katalogu
Uruchom _setup_lesson.sh_ z opcją _06_, aby wypakować do katalogu projektu wszystkie pliki z 6. części tutorialu.
```bash
chmod +x setup_lesson.sh
sh setup_lesson.sh 06
```