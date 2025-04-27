import torch

# Sprawdź, czy masz dostęp do GPU
print(torch.cuda.is_available())  # Powinno zwrócić True, jeśli GPU jest dostępne

# Możesz teraz używać torch w całym skrypcie
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')