# Dokumentacja komponentów frontendowych (React)

---

## Ogólny opis

Aplikacja frontendowa została zbudowana w React i składa się z komponentów o ograniczonej logice, w których zależności są minimalizowane. Wszystkie komponenty (z wyjątkiem nawigacji) są zależne jedynie od `AuthContext`, dzięki czemu są łatwe do wymiany lub rozszerzenia.

Każdy komponent pełni pojedynczą funkcję, co znacząco ułatwia testowanie i rozwój. Jedynym wyjątkiem jest `NavbarComponent`, który ze względu na swoją specyfikę (nawigacja + obsługa wylogowania i przekazywanie zapytań typu google-auth do contextu) realizuje więcej niż jedną funkcję, ale korzysta z kontekstu uwierzytelniania, aby uprościć integrację.

---

## Cechy architektury

- **Izolowana logika komponentów:** Komponenty nie zawierają zbędnych zależności poza `AuthContext`.
- **Zamienność komponentów:** Komponenty można łatwo wymieniać dzięki ograniczonej integracji wewnętrznej.
- **Zasada Open-Closed:** Trudno jest modyfikować istniejące endpointy REST oraz logike komponentów, ale łatwo dodawać nowe, co wspiera rozwój bez ryzyka regresji.
- **Zasada Dependency Inversion:** Komunikacja z backendem odbywa się wyłącznie przez REST API, a nie poprzez bezpośrednie importy z Django, backend to czarna skrzynka z perspektywy fronta.
- **Zasada Liskov:** Ze względu na charakter Reacta (brak dziedziczenia interfejsów) LSP nie ma zastosowania na froncie, ale jest stosowana po stronie backendu.

---

## Opis komponentów

### 1. App.jsx
Główny komponent. Renderuje nawigację i trasy oraz inicjalizuje konteksty.

### 2. AuthContext.jsx
Zarządza sesją użytkownika i stanem uwierzytelnienia. Komunikuje się z backendem przez REST API (`/api/auth-status`, `/api/logout`, itd.). Zapewnia cookies podczas wszystkich requestów.

### 3. NavbarComponent.jsx
Renderuje pasek nawigacyjny oraz obsługuje wylogowanie i przesyłanie danych z logowania do contextu.

### 4. UploadPage.jsx
Obsługuje logikę przesyłania plików oraz walidację.

### 5. ContactPage.jsx
Wyświetla formularz kontaktowy i przesyła dane do backendu.

### 6. AdminPage.jsx
Panel administracyjny, umożliwia przesyłanie ustawień do backendu.

### 7. ResultsPage.jsx
Wyświetla dane wynikowe i umożliwia ich eksport.

### 8. ProfileComponent.jsx
Pobiera i wyświetla obrazy przypisane do profilu użytkownika.

---

Struktura została zaprojektowana tak, aby była łatwa w utrzymaniu, rozwoju i testowaniu bez wprowadzania głębokich zależności pomiędzy komponentami.