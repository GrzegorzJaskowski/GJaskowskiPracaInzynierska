# System Automatyzujący Egzaminację

## Wymagania

- Python 3.12 lub nowszy
- Poetry
- Klucz API OpenAI
- Docker i Docker Compose
- Tesseract OCR

## Instalacja zależności

- Instalacja Pythona (>=3.12)

  - Pobierz instalator ze strony: https://www.python.org/downloads/
  - Podczas instalacji zaznacz opcję "Add Python to PATH"

- Tesseract OCR

  - Windows: Pobierz instalator ze strony: https://github.com/UB-Mannheim/tesseract/wiki
  - Zainstaluj w domyślnej lokalizacji: `C:\Program Files\Tesseract-OCR\`
  - Alternatywnie możesz zmienić ścieżkę w pliku `.env` (zmienna `TESSERACT_PATH`)

- Klonowanie repozytorium

```bash
git clone
```

- Instalacja poetry

```bash
pip install poetry
```

- Instalacja zależności

```bash
poetry install
```

## Konfiguracja zmiennych środowiskowych

W pliku `.env` utworzonym w głównym katalogu projektu należy umieścić zmienne na wzór `.env.example`:

```bash
# Klucz pozwalający na korzystanie z API modeli OpenAI
OPENAI_API_KEY=sk-your-api-key-here

# Zmienna określająca czy pytania egzaminacyjne powinny być każdorazowo generowane
GENERATE_QUESTIONS=false

# Ścieżka instalacyjna tesseract
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe

# Ścieżka do pliku PDF z którego jest tworzona baza wiedzy
PDF_PATH=src/praca/data/data.pdf

# Ścieżka w której jest zapisywana baza wiedzy
DB_PATH=faiss_db

# Liczba pytań w egzaminie
QUESTION_NUMBER=5
```

## Uruchomienie aplikacji

- Uruchomienie OpenWebUI na `http://localhost:3000`

```bash
docker-compose up -d
```

- Uruchomienie serwera uvicorn na `http://localhost:8000`

```bash
poetry run python -m praca
```

- Konfiguracja OpenWebUI:
  - Otwórz http://localhost:3000
  - Zarejestruj się
  - Kliknij na ikonkę ikonkę z inicjałami użytkownika w prawym górnym roku
  - Otwórz zakładkę `Settings` i wybierz kategorię `Connections`
  - Dodaj nowe połącznie:
    - URL: `http://localhost:8000/v1`
    - Name: `Egzaminator`
    - Key: `dummy-key` - chat łączy się z lokalnym serwerem, a klucz dostępu jest obsługiwany z poziomu systemu
