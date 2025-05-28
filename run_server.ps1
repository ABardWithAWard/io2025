# Change to frontend directory and build React app
Set-Location -Path "frontend"
Write-Host "Building React app..."
($env:HTTPS = "true") -and (npm run build)

# Change back to Django directory
Set-Location -Path "..\ocr"
Write-Host "Collecting static files..."
python manage.py collectstatic --noinput

# Start Django server
Write-Host "Starting Django server..."
Set-Location -Path ".."
python ocr/manage.py runserver_plus --cert-file cert.pem --key-file key.pem 0.0.0.0:8000
