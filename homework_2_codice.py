#Petrina Gaia - s308033, Ruggiero Alessandro - s309582, Ruscitto Virginia - s307256

#!pip3 install numpy matplotlib librosa tqdm numba

import ipywidgets as widgets
from IPython.display import display,Audio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm

BRANO_ROCK = 'Nirvana - Smells Like Teen Spirit (Lyrics).wav'
BRANO_CLASSICO = 'Edvard Grieg Peer Gynt - Morning Mood.wav'

def apri_file (nome_file: str,  inizio = 1*60+6, durata = 22) -> tuple[np.ndarray,int|float] :
  y, sr = librosa.load(nome_file, offset = inizio, duration = durata, sr = None)
  if len(y.shape) == 1:
    print ("Il segnale è mono, ha lunghezza",y.shape[0])
    if y.shape[0] != sr*durata:
      raise Exception(f"Problemi con la durata file sr:{sr}, dimensioni: {y.shape[0]}, {durata=}")
  else:
    raise Exception("Il file non è mono")
  print ("Frequenza di campionamento: ", sr, "Hz")
  plt.figure(figsize=(14, 5))
  librosa.display.waveshow(y, sr=sr)
  plt.title("Forma d'onda")
  return y, sr
  
def trova_T(sr: int):
  best_T = None
  best_f = None
  n = np.random.normal(0, 1, 16384)
  normalizzazione = np.sqrt(1/16384)
  n = n * normalizzazione
  T_valori = np.linspace(0.0001, 0.001, 1000)
  for T in T_valori:
    t, h = rect_t(T, sr)
    y = np.convolve (n, h)
    spettro_y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    spettro_y=spettro_y/max(spettro_y)
    ampiezza_segnale_db = 10 * np.log10(np.maximum(spettro_y, 1e-10))
    freq_segnale_convoluto = np.fft.fftshift(np.fft.fftfreq(len(y), 1 / sr))
    idx_3db = np.where(np.abs(ampiezza_segnale_db + 3) <=0.1)[0] #perchè restituisce una tupla
    if len(idx_3db) > 0:
      ultima_freq = freq_segnale_convoluto[idx_3db[-1]]
      if np.abs(ultima_freq - 1000) < 50 and (best_f is None or best_f > np.abs(ultima_freq - 1000)):
        best_T = T
        best_f = np.abs(ultima_freq - 1000)


  if best_T is not None:
      t, h = rect_t(best_T, sr_classico)
      y = np.convolve(n, h)
      spettro_y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
      spettro_y = spettro_y / max(spettro_y)
      ampiezza_segnale_db = 10 * np.log10(np.maximum(spettro_y, 1e-10))
      freq_segnale_convoluto = np.fft.fftshift(np.fft.fftfreq(len(y), 1 / sr_classico))
      plt.plot(freq_segnale_convoluto, ampiezza_segnale_db)
      plt.axhline(y=-3, color='red', alpha=0.2)
      plt.axvline(x=1000, color='red', alpha=0.2)
      plt.title('Scelta del parametro T')
      plt.xlabel('Frequenza (Hz)')
      plt.ylabel('Densità Spettrale di Energia (dB)')

  return best_T

def rect_t(T:float, sr: int):
    n_campioni = int(T * sr)
    t = np.linspace(0, T, n_campioni)
    p = np.ones(n_campioni)
    normalizzazione = 1/n_campioni
    p = p * normalizzazione
    return t,p
    
def trova_B(sr: int):
  best_B = None
  best_f = None
  n = np.random.normal(0, 1, 16384)
  normalizzazione = np.sqrt(1/16384)
  n = n * normalizzazione
  B_valori = np.linspace(950, 1050, 500)
  for B in B_valori:
    t, h = rect_f(B, sr)
    y = np.convolve (n, h)
    spettro_y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    spettro_y=spettro_y/max(spettro_y)
    ampiezza_segnale_db = 10 * np.log10(np.maximum(spettro_y, 1e-10))
    freq_segnale_convoluto = np.fft.fftshift(np.fft.fftfreq(len(y), 1 / sr))
    idx_3db = np.where(np.abs(ampiezza_segnale_db + 3) <=0.1)[0] #perchè restituisce una tupla
    if len(idx_3db) > 0:
      ultima_freq = freq_segnale_convoluto[idx_3db[-1]]
      if np.abs(ultima_freq - 1000) < 50 and (best_f is None or best_f > np.abs(ultima_freq - 1000)):
        best_B = B
        best_f = np.abs(ultima_freq - 1000)


  if best_B is not None:
      t, h = rect_f(best_B, sr)
      y = np.convolve(n, h)
      spettro_y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
      spettro_y = spettro_y / max(spettro_y)
      ampiezza_segnale_db = 10 * np.log10(np.maximum(spettro_y, 1e-10))
      freq_segnale_convoluto = np.fft.fftshift(np.fft.fftfreq(len(y), 1 / sr))
      plt.plot(freq_segnale_convoluto, ampiezza_segnale_db)
      plt.axhline(y=-3, color='red', alpha=0.2)
      plt.axvline(x=1000, color='red', alpha=0.2)
      plt.title('Scelta del parametro B')
      plt.xlabel('Frequenza (Hz)')
      plt.ylabel('Densità Spettrale di Energia (dB)')

  return best_B

def rect_f (B:float, sr:int, L:int=None):
  if L is None:
    L = 0.0015/(B*2/sr) # deciso empiricamente 1/(B*2/sr) è la frazione di ciò che passa
  N = int(L * sr)
  x = np.arange(-N//2,N//2+1)/sr
  h = 2 * B * np.sinc (x * 2 * B)
  normalizzazione = 1/sr
  h = h*normalizzazione
  return x+L/2,h
  
def passa_alto (B:float, sr:int, T:int=None):
  t,h = rect_f(B, sr)
  indice_max = np.argmax(h)
  h = h * (-1)
  h[indice_max] = h[indice_max] + 1
  return t, h

def mostra_grafici_convoluzione(segnale: np.ndarray,
                                y: np.ndarray,
                                sr: int,
                                supporto_h: np.ndarray = None,
                                h: np.ndarray = None,
                                ampiezza_grafico_frequenze: float | None = None,
                                sezione_segnale_ingrandita: tuple[float, float] | None = None,
                                frequenza_di_taglio: float | None = None,
                                ):
    grafico_n = 0
    righe = 4
    if supporto_h is not None:
        righe += 1
    if ampiezza_grafico_frequenze is not None:
        righe += 2
    if sezione_segnale_ingrandita is not None:
        righe += 2

    fig, axes = plt.subplots(nrows=righe, ncols=1, figsize=(12, 4 * righe))
    supporto_segnale = np.arange(0, len(segnale)) / sr
    # Grafico segnale in ingresso
    axes[grafico_n].plot(supporto_segnale, segnale)
    axes[grafico_n].set_title("Segnale in ingresso")
    axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
    axes[grafico_n].set_xlabel ("Tempo (s)")
    axes[grafico_n].set_ylabel ("Ampiezza segnale")
    grafico_n += 1

    # Grafico segnale in uscita
    x = np.arange(0, len(y)) / sr
    axes[grafico_n].plot(x, y)
    axes[grafico_n].set_title("Segnale in uscita")
    axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
    axes[grafico_n].set_xlabel ("Tempo (s)")
    axes[grafico_n].set_ylabel ("Ampiezza segnale")
    grafico_n += 1

    # Grafico risposta all’impulso del filtro, se fornito
    if supporto_h is not None and h is not None:
        axes[grafico_n].scatter(supporto_h, h)
        axes[grafico_n].set_title("Risposta all’impulso del filtro")
        axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
        axes[grafico_n].set_xlabel ("Tempo (s)")
        axes[grafico_n].set_ylabel ("Ampiezza")
        grafico_n += 1

    # Spettro segnale in ingresso
    ampiezza_segnale = np.abs(np.fft.fftshift(np.fft.fft(segnale))) ** 2
    ampiezza_segnale_db = 10 * np.log10(np.maximum(ampiezza_segnale, 1e-20))
    freq_segnale = np.fft.fftshift(np.fft.fftfreq(len(segnale), 1 / sr))
    axes[grafico_n].plot(freq_segnale, ampiezza_segnale_db)
    axes[grafico_n].set_title("Spettro del segnale in ingresso")
    axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
    axes[grafico_n].set_xlabel ("Frequenza (Hz)")
    axes[grafico_n].set_ylabel ("Densità Spettrale di Energia (dB)")
    if frequenza_di_taglio is not None:
      axes[grafico_n].axvline(x=frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
      axes[grafico_n].axvline(x=-frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
    grafico_n += 1

    # Spettro segnale in uscita
    ampiezza_segnale_convoluto = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    ampiezza_segnale_convoluto_db = 10*np.log10(np.maximum(ampiezza_segnale_convoluto, 1e-20))
    freq_segnale_convoluto = np.fft.fftshift(np.fft.fftfreq(len(y), 1 / sr))
    axes[grafico_n].plot(freq_segnale_convoluto, ampiezza_segnale_convoluto_db)
    axes[grafico_n].set_title("Spettro segnale in uscita")
    axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
    axes[grafico_n].set_xlabel ("Frequenza (Hz)")
    axes[grafico_n].set_ylabel ("Densità Spettrale di Energia (dB)")
    if frequenza_di_taglio is not None:
      axes[grafico_n].axvline(x=frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
      axes[grafico_n].axvline(x=-frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
    grafico_n += 1

    # Zoom dello spettro, se richiesto
    if ampiezza_grafico_frequenze is not None:
        centro_spettro = np.argmin(np.abs(freq_segnale))
        freq_segnale_zoom = freq_segnale[centro_spettro - int(ampiezza_grafico_frequenze): centro_spettro + int(ampiezza_grafico_frequenze)]
        ampiezza_segnale_zoom = ampiezza_segnale_db[centro_spettro - int(ampiezza_grafico_frequenze): centro_spettro + int(ampiezza_grafico_frequenze)]
        freq_segnale_convoluto_zoom = freq_segnale_convoluto[centro_spettro - int(ampiezza_grafico_frequenze): centro_spettro + int(ampiezza_grafico_frequenze)]
        ampiezza_segnale_convoluto_zoom = ampiezza_segnale_convoluto_db[centro_spettro - int(ampiezza_grafico_frequenze): centro_spettro + int(ampiezza_grafico_frequenze)]

        axes[grafico_n].plot(freq_segnale_zoom, ampiezza_segnale_zoom)
        axes[grafico_n].set_title("Spettro segnale in ingresso ingrandito")
        axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
        axes[grafico_n].set_xlabel ("Frequenza (Hz)")
        axes[grafico_n].set_ylabel ("Densità Spettrale di Energia (dB)")
        if frequenza_di_taglio is not None:
          axes[grafico_n].axvline(x=frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
          axes[grafico_n].axvline(x=-frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
        grafico_n += 1

        axes[grafico_n].plot(freq_segnale_convoluto_zoom, ampiezza_segnale_convoluto_zoom)
        axes[grafico_n].set_title("Spettro segnale in uscita ingrandito")
        axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
        axes[grafico_n].set_xlabel ("Frequenza (Hz)")
        axes[grafico_n].set_ylabel ("Densità Spettrale di Energia (dB)")
        if frequenza_di_taglio is not None:
          axes[grafico_n].axvline(x=frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
          axes[grafico_n].axvline(x=-frequenza_di_taglio, color='r', label='Frequenza di Taglio', alpha=0.2)
        grafico_n += 1

    # Ingrandimento del segnale, se richiesto
    if sezione_segnale_ingrandita is not None:
        inizio, fine = sezione_segnale_ingrandita
        segnale_ingrandito = segnale[int(inizio * sr):int(fine * sr)]
        supporto_segnale_ingrandito = supporto_segnale[int(inizio * sr):int(fine * sr)]
        y_ingrandita = y[int(inizio * sr):int(fine * sr)]
        supporto_segnale_convoluto_ingrandito = x[int(inizio * sr):int(fine * sr)]

        axes[grafico_n].plot(supporto_segnale_ingrandito, segnale_ingrandito)
        axes[grafico_n].set_title("Segnale in ingresso ingrandito")
        axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
        axes[grafico_n].set_xlabel ("Tempo (s)")
        axes[grafico_n].set_ylabel ("Ampiezza segnale")
        grafico_n += 1

        axes[grafico_n].plot(supporto_segnale_convoluto_ingrandito, y_ingrandita)
        axes[grafico_n].set_title("Segnale in uscita ingrandito")
        axes[grafico_n].grid(True, linestyle='--', linewidth=0.5, which='both')
        axes[grafico_n].set_xlabel ("Tempo (s)")
        axes[grafico_n].set_ylabel ("Ampiezza segnale")
        grafico_n += 1

    plt.tight_layout()
    plt.show()

def funzione_trasferimento (risposta_impulso: np.array, sr:int):
  n = np.random.normal(0, 1, 2**14)
  normalizzazione = np.sqrt(1/2**14)
  n = n * normalizzazione
  y = np.convolve (n, risposta_impulso)
  spettro_y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
  ampiezza_segnale_db = 10 * np.log10(np.maximum(spettro_y, 1e-10))
  freq_segnale_convoluto = np.fft.fftshift(np.fft.fftfreq(len(y), 1 / sr))
  plt.plot(freq_segnale_convoluto, ampiezza_segnale_db)
  plt.title('Funzione di trasferimento')
  plt.xlabel('Frequenza (Hz)')
  plt.ylabel('Densità Spettrale di Energia (dB)')

def main():
  print("HOMEWORK 2 TES\n\n\n")
  
  
  print("\n\nBRANO CLASSICO")
  
  brano_classico, sr_classico = apri_file(BRANO_CLASSICO)
  display(Audio(data=brano_classico, rate=sr_classico))
  
  #h1
  T=trova_T(sr_classico)
  print(T)
  t, h = rect_t(T, sr_classico)
  segnale_convoluto = np.convolve (brano_classico, h)
  mostra_grafici_convoluzione(
    brano_classico,
    segnale_convoluto,
    sr_classico,
    t,
    h,
    ampiezza_grafico_frequenze= 3000,
    sezione_segnale_ingrandita=(0,0.1)
  )
  funzione_trasferimento (h, sr_classico)
  display(Audio(data=segnale_convoluto, rate=sr_classico))
  
  #h2
  B = trova_B(sr_classico)
  print(B)
  
  t,h = rect_f(B, sr_classico)
  segnale_convoluto = np.convolve (brano_classico, h)
  mostra_grafici_convoluzione(
    brano_classico,
    segnale_convoluto,
    sr_classico,
    t,
    h,
    ampiezza_grafico_frequenze= 3000,
    sezione_segnale_ingrandita=(0,0.1),
    frequenza_di_taglio=B
  )
  funzione_trasferimento (h, sr_classico)
  display(Audio(data=segnale_convoluto, rate=sr_classico))
  
  #h3
  t, h = passa_alto (B, sr_classico)
  segnale_convoluto = np.convolve (brano_classico, h)
  mostra_grafici_convoluzione(
    brano_classico,
    segnale_convoluto,
    sr_classico,
    t,
    h,
    ampiezza_grafico_frequenze= 3000,
    sezione_segnale_ingrandita=(0,0.1),
    frequenza_di_taglio=B
  )
  funzione_trasferimento (h, sr_classico)
  display(Audio(data=segnale_convoluto, rate=sr_classico))


  print("\n\nBRANO ROCK")
  
  brano_rock, sr_rock = apri_file(BRANO_ROCK)
  display(Audio(data=brano_rock, rate=sr_rock))
  
  #h1
  t, h = rect_t(T, sr_rock)
  segnale_convoluto = np.convolve (brano_rock, h)
  mostra_grafici_convoluzione(
    brano_rock,
    segnale_convoluto,
    sr_rock,
    t,
    h,
    ampiezza_grafico_frequenze= 3000,
    sezione_segnale_ingrandita=(0,0.1)
  )
  display(Audio(data=segnale_convoluto, rate=sr_rock))
  
  #h2
  t,h = rect_f(B, sr_rock)
  segnale_convoluto = np.convolve (brano_rock, h)
  mostra_grafici_convoluzione(
    brano_rock,
    segnale_convoluto,
    sr_rock,
    t,
    h,
    ampiezza_grafico_frequenze= 3000,
    sezione_segnale_ingrandita=(0,0.1),
    frequenza_di_taglio=B
  )
  funzione_trasferimento (h, sr_rock)
  display(Audio(data=segnale_convoluto, rate=sr_rock))
  
  #h3
  t, h = passa_alto (B, sr_rock)
  segnale_convoluto = np.convolve (brano_rock, h)
  mostra_grafici_convoluzione(
    brano_rock,
    segnale_convoluto,
    sr_rock,
    t,
    h,
    ampiezza_grafico_frequenze= 3000,
    sezione_segnale_ingrandita=(0,0.1),
    frequenza_di_taglio=B
  )
  funzione_trasferimento (h, sr_rock)
  display(Audio(data=segnale_convoluto, rate=sr_rock))
  
  
