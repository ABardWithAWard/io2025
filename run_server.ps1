# Change to frontend directory and build React app
Set-Location -Path "frontend"
Write-Host "Building React app..."
npm run build

# Change back to Django directory
Set-Location -Path "..\ocr"
Write-Host "Collecting static files..."
python manage.py collectstatic --noinput

# Start Django server
Write-Host "Starting Django server..."
python manage.py runserver 