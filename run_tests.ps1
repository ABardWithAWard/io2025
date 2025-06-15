# 1. Uruchomienie frontend (React) w nowym oknie
Write-Host "🚀 Uruchamianie frontend (React)..."
Start-Process powershell -ArgumentList 'npm start' -WorkingDirectory "$PWD\frontend"

# 2. Poczekaj chwilę, żeby frontend się uruchomił
Start-Sleep -Seconds 5

# 3. Przejście do Django i uruchomienie testów
Set-Location -Path "ocr"
Write-Host "Uruchamianie testów Django..."
coverage erase
coverage run --rcfile=.coveragerc manage.py test
coverage html

# 4. Otwórz raport pokrycia w przeglądarce
Write-Host "Otwieranie raportu pokrycia..."
Start-Process "htmlcov\index.html"