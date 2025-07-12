# ssya

SSYA to graficzne narzędzie do segmentacji i wyszukiwania podobnych regionów w zbiorach obrazów, wykorzystujące Segment Anything v2 (SAM2).

## Funkcje
- Offline’owe indeksowanie i cache’owanie embeddingów z SAM2
- Interaktywne GUI z paskami postępu i filtrowaniem według progu podobieństwa
- Szybkie wyszukiwanie podobnych wykryć

## Wymagania
- Python 3.11+

## Instalacja
```bash
git clone <repo-url>
cd ssya
pdm install
```

## Użycie
```bash
ssya -i /ścieżka/do/dataset
```
Jeśli nie podasz `-i`, pojawi się okno dialogowe do wyboru folderu ze zbiorami.

## Format danych
Umieść w katalogu obrazy i pliki anotacji TXT (jedna linia na obiekt: `klasa xc yc szerokość wysokość`, wartości znormalizowane).

## Testy
```bash
pdm run pytest
```

## Wydania
Nowe tagi `vX.X.X` wrzucane na `main` są automatycznie publikowane na PyPI.  
Możesz też ręcznie:
```bash
git tag vX.X.X
git push --tags
pdm build
pdm publish
```

