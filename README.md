# Personalny HR Asystent z AI
 # Problem
Rekruterzy i kandydaci mają trudności z szybką i obiektywną oceną CV. Tradycyjne metody są czasochłonne, kosztowne i często nieefektywne.

 # Rozwiązanie
Projekt łączy klasyczny model ML oraz nowoczesne AI, aby:

 - Automatycznie ocenić jakość CV na skali 1–5

 - Wygenerować praktyczne porady dotyczące ulepszenia dokumentu

 - Obsługiwać pliki PDF i analizować ich treść w czasie rzeczywistym

 # Technologie
- TF-IDF (TfidfVectorizer) — przekształca tekst CV w wektory cech (bag-of-words z uwzględnieniem ważności słów)

- RandomForestRegressor (scikit-learn) — regresyjna ocena jakości CV na podstawie wektorów TF-IDF

- GPT-4o (OpenAI API) — generowanie jasnych i spersonalizowanych wskazówek na podstawie treści CV

- Streamlit — prosty interfejs do użytku webowego

- pdfplumber — ekstrakcja tekstu z plików PDF

 # Dwa tryby użycia
- Enterprise: masowe ocenianie CV i wsparcie dla HR/rekruterów

- Personal: indywidualna ocena jednego CV z podpowiedziami

 # Efekt
Szybka, rzetelna i zrozumiała analiza CV — zarówno dla firm, jak i osób indywidualnych. Projekt łączy klasyczne uczenie maszynowe (ML) z generatywną AI.
