#Petrina Gaia - s308033, Ruggiero Alessandro - s309582, Ruscitto Virginia - s307256

#!pip3 install numpy matplotlib librosa tqdm numba

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
from numba import njit

BRANO_ROCK = 'Nirvana - Smells Like Teen Spirit (Lyrics).wav'
BRANO_CLASSICO = 'Edvard Grieg Peer Gynt - Morning Mood.wav'

def apri_file (nome_file: str, M: int | float, inizio = 1*60+6, durata = 22) -> tuple[np.ndarray,int|float] :
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
  plt.show()
  campioni_per_sezione = int(M*sr)
  n_sequenze = y.shape[0] // campioni_per_sezione
  y = y[:n_sequenze * campioni_per_sezione] # si getta via la parte restante
  y = y.reshape((n_sequenze, campioni_per_sezione))
  return y, sr
  
# versione standard
def dft_old (segnale: np.ndarray, use_tqdm:bool) -> np.ndarray:
  N = len(segnale)
  X = np.zeros (N, dtype=np.complex_)
  if use_tqdm:
    r = tqdm(range(N))
  else:
    r = range(N)
  for k in r:
    for n  in range (N):
      X[k] += segnale[n] * np.exp(-2j * np.pi * n * k / N)
  return X

# versione compilata
@njit(parallel=False)
def dft_opt (segnale: np.ndarray) -> np.ndarray:
  N = len(segnale)
  X = np.zeros (N, dtype=np.complex_)
  for k in range(N):
    for n  in range (N):
      X[k] += segnale[n] * np.exp(-2j * np.pi * n * k / N)
  return X
  
def processa_sottofinestra (x:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  fft = np.fft.fftshift(np.fft.fft(x))
  dft = np.fft.fftshift(dft_opt(x))
  spettro_dft = np.abs(dft)**2
  spettro_fft = np.abs(fft)**2
  return dft, fft, spettro_dft, spettro_fft

def scala_sottofinestra (spettro:np.ndarray, sr: int | float) -> tuple[np.ndarray, np.ndarray]:
  frequencies = np.fft.fftfreq(spettro.shape[0], d=1/sr)
  frequencies_shifted = np.fft.fftshift(frequencies)
  frequencies_kHz = frequencies_shifted / 1000 
  return frequencies_kHz, 10*np.log10(spettro)
  
def main():
  print("HOMEWORK 1 TES\n\n\n")

  M = float(input("Inserisci la durata delle sotto-finestre in secondi (in intero o in decimale): "))
  if M.is_integer():
    M = int(M)

  print("\n\nBRANO CLASSICO")
  y1,sr1 = apri_file (BRANO_CLASSICO,M,durata=22)
  print(y1.shape)
  finestre_processate1:list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
  for i in tqdm(range (y1.shape[0])):
    r = processa_sottofinestra(y1[i])
    finestre_processate1.append(r)
  fig, axis = plt.subplots(len (finestre_processate1), 2, figsize=(15, M*200))
  fig.text(0.25, 1, 'SPETTRO DFT', fontsize=16, ha='center')
  fig.text(0.75, 1, 'SPETTRO FFT', fontsize=16, ha='center')
  plt.tight_layout(rect=[0, 0, 1, 0.995], h_pad=6, w_pad=1.8)
  for i in range (len (finestre_processate1)):
    dft, fft, spettro_dft, spettro_fft = finestre_processate1[i]
    errore = np.mean(np.square(spettro_fft - spettro_dft))
    x_dft, y_dft = scala_sottofinestra(spettro_dft, sr1)
    x_fft, y_fft = scala_sottofinestra(spettro_fft, sr1)
    axis[i][0].plot(x_dft, y_dft, color = "r")
    axis[i][1].plot(x_fft, y_fft, color = "b")
    axis[i][0].set_title(f"DFT finestra {i*M}-{(i+1)*M} secondi (Errore: {errore:.2e})")
    axis[i][1].set_title(f"FFT finestra {i*M}-{(i+1)*M} secondi")
    for a in axis [i]:
      a.set_xlabel ("Frequenza (kHz)")
      a.set_ylabel ("Densità Spettrale di Energia (dB)")
      a.grid(True, linestyle='--', linewidth=0.5, which='both')

  plt.show()

  print("\n\nBRANO ROCK")
  y2,sr2 = apri_file (BRANO_ROCK,M,durata=22)
  print(y2.shape)
  finestre_processate2:list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
  for i in tqdm(range (y2.shape[0])):
    r = processa_sottofinestra(y2[i])
    finestre_processate2.append(r)
  fig, axis = plt.subplots(len (finestre_processate2), 2, figsize=(15, M*200))
  fig.text(0.25, 1, 'SPETTRO DFT', fontsize=16, ha='center')
  fig.text(0.75, 1, 'SPETTRO FFT', fontsize=16, ha='center')
  plt.tight_layout(rect=[0, 0, 1, 0.995], h_pad=6, w_pad=1.8)
  for i in range (len (finestre_processate2)):
    dft, fft, spettro_dft, spettro_fft = finestre_processate2[i]
    errore = np.mean(np.square(spettro_fft - spettro_dft))
    x_dft, y_dft = scala_sottofinestra(spettro_dft, sr2)
    x_fft, y_fft = scala_sottofinestra(spettro_fft, sr2)
    axis[i][0].plot(x_dft, y_dft, color = "r")
    axis[i][1].plot(x_fft, y_fft, color = "b")
    axis[i][0].set_title(f"DFT finestra {i*M}-{(i+1)*M} secondi (Errore: {errore:.2e})")
    axis[i][1].set_title(f"FFT finestra {i*M}-{(i+1)*M} secondi")
    for a in axis [i]:
      a.set_xlabel ("Frequenza (kHz)")
      a.set_ylabel ("Densità Spettrale di Energia (dB)")
      a.grid(True, linestyle='--', linewidth=0.5, which='both')

  plt.show()

  print("\n\nCONFRONTO")
  fig, axis = plt.subplots(len (finestre_processate1), 1, figsize=(15, M*200))
  fig.text(0.5, 1, 'SPETTRI FFT', fontsize=16, ha='center')
  plt.tight_layout(rect=[0, 0, 1, 0.995], h_pad=6, w_pad=1.8)
  for i in range (len (finestre_processate1)):
    spettro_fft_1 = finestre_processate1[i][3]
    spettro_fft_2 = finestre_processate2[i][3]
    x_fft_1, y_fft_1 = scala_sottofinestra(spettro_fft_1, sr1)
    x_fft_2, y_fft_2 = scala_sottofinestra(spettro_fft_2, sr2)
    axis[i].plot(x_fft_1, y_fft_1, color = "pink", label="classica", alpha=0.7)
    axis[i].plot(x_fft_2, y_fft_2, color = "green", label="rock", alpha=0.3)
    axis[i].set_title(f"FFT finestra {i*M}-{(i+1)*M} secondi")
    axis[i].set_xlabel ("Frequenza (kHz)")
    axis[i].set_ylabel ("Densità Spettrale di Energia (dB)")
    axis[i].legend(
      loc='upper right',
      shadow=True,
      fontsize='small',
      title_fontsize='medium',
    )
    axis[i].grid(True, linestyle='--', linewidth=0.5, which='both')
  
  plt.show()
