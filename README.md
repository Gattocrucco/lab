# lab.py

Piccolo modulo con funzioni python per il laboratorio di fisica:

* fit
* formattazione numeri con incertezze
* calcolo errore dei multimetri

Requisiti: **scipy**

**NON TESTATO SU PYTHON 2**

## Versioni

### 2016.11

**questa versione rompe la compatibilità**

* migliorate funzioni di formattazione (supportano esponente comune e notazione compatta)
* funzione num2si che formatta con i suffissi SI (k, m, etc.)
* unificate funzioni di fit lineare nella funzione `fit_linear()`, che calcola anche la covarianza
* aggiunta testsuite
