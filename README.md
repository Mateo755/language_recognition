# Language detection - model deployment pipeline

## Konfiguracja środowiska (setup)

### Wymagania

- **Python 3.10+** (zgodnie z `pyproject.toml`)
- Katalog główny repozytorium (tam, gdzie leży `pyproject.toml`)

### Procedura

1. **Przejdź do katalogu projektu**

   ```bash
   cd /ścieżka/do/language_recognition
   ```

2. **Utwórz wirtualne środowisko** (przykład: folder `.venv` w projekcie)

   ```bash
   python3 -m venv .venv
   ```

3. **Aktywuj venv**

   - Linux / macOS:

     ```bash
     source .venv/bin/activate
     ```

   - Windows (cmd): `.venv\Scripts\activate.bat`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`

4. **(Opcjonalnie) Zaktualizuj pip** w tym venv — mniej problemów przy instalacji z `pyproject.toml`:

   ```bash
   python -m pip install -U pip
   ```

5. **Zainstaluj projekt w trybie edytowalnym** — zależności biorą się z `pyproject.toml`, pakiety `src` i `scripts` stają się importowalne:

   ```bash
   pip install -e .
   ```

   Równoważnie:

   ```bash
   pip install -r requirements.txt
   ```

6. **Dane** — domyślnie pipeline oczekuje pliku **`data/language_detection.csv`** z kolumnami `Text` i `Language`. Upewnij się, że plik istnieje (lub dostosuj ścieżkę w `scripts/train_pipeline.py`).

### Uruchomienie treningu

Po setupie, z aktywnym venv i z katalogu projektu (dowolna z opcji):

```bash
train-pipeline
```

```bash
python -m scripts.train_pipeline
```

```bash
python scripts/train_pipeline.py
```

Wytrenowany model zapisuje się domyślnie do **`models/trained_pipeline-0.1.0.pkl`** (folder `models/` utworzy się przy zapisie, jeśli go nie ma).

---

## Struktura repozytorium

Katalog główny projektu (np. `language_recognition/`):

```text
language_recognition/
├── app/                    # serwer i logika API (FastAPI)
├── data/
│   └── language_detection.csv
├── models/
│   └── trained_pipeline-0.1.0.pkl   # wyjście treningu (opcjonalnie w repo)
├── scripts/
│   ├── __init__.py
│   └── train_pipeline.py   # pipeline treningowy
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── data_split.py
│   └── model_training.py
├── pyproject.toml          # pakiet, zależności, entry point `train-pipeline`
├── requirements.txt        # `-e .` → instalacja z pyproject.toml
├── README.md
└── .gitignore
```

Po `pip install -e .` w katalogu głównym pojawia się lokalnie `*.egg-info/` (jest w `.gitignore`).
