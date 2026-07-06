## Homework 1 - Analisi in frequenza (DFT vs FFT)
Analisi in frequenza di due brani musicali campionati ("Smells Like Teen Spirit" dei Nirvana e "Morning Mood" di Grieg), confrontando lo spettro di energia ottenuto tramite:
- DFT implementata manualmente da zero
- FFT tramite libreria (NumPy)
I brani vengono divisi in sotto-finestre temporali di durata configurabile (parametro `M`) e per ciascuna finestra viene calcolato e visualizzato lo spettro.
 
## Homework 2 - Filtraggio di segnali audio
Applicazione di tre filtri lineari ai due brani musicali tramite convoluzione discreta:
1. Filtro passa-basso (porta nel tempo)
2. Filtro passa-basso (porta in frequenza, tramite sinc)
3. Filtro passa-alto (complementare al secondo)
Per ciascun filtro vengono confrontati il segnale e lo spettro in ingresso e in uscita, stimando inoltre la funzione di trasferimento tramite rumore bianco gaussiano come segnale di test.

---

### Requisiti
 
```bash
pip3 install numpy matplotlib librosa tqdm numba
```

Per l'homework 2 è necessario anche `ipywidgets` (uso pensato per Jupyter/Colab).
 
### Esecuzione
 
```bash
python3 homework1.py
```
Verrà chiesto di inserire il parametro `M`, che rappresenta la durata delle sotto-finestre temporali in secondi.
 
Per l’homework 2 è consigliato utilizzare un ambiente Jupyter/Colab per poter ascoltare l’audio filtrato.
 
### Note
 
I file audio utilizzati non sono inclusi nel repository per motivi di copyright.
Devono essere aggiunti manualmente nella stessa cartella degli script mantenendo gli stessi nomi dei file originali