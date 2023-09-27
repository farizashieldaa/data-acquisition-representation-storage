import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile 
import librosa
import matplotlib.pyplot as plt

# Menentukan sampling rate
sr=44100
# Menentukan frequency dalam satu detik
freq=1
# Menentukan time (sumbu x)
length=1

# Membuat fungsi linear dari nol s/d time dengan titik berjumlah sr
# 1/sr --> stepsize, karena kita mau ada sr titik per detik
t=np.arange(0, length, 1.0/sr)
# Membuat wave dengan sin function wave = A*sin(2*pi*freq*t)
signal=np.sin(np.pi*2*freq*t)

# Range satu cycle dalam satu detik, range 1 s/d -1, starting point 0
plt.plot(t, signal)
plt.show()

# Menyimpan sinyal gelombang ke dalam format wav
wavfile.write("file.wav", sr, signal)


# Mengubah frequensi agar terdengar manusia
freq = 200
signal2 = np.sin(np.pi*2*freq*t)
plt.plot(t, signal2)
plt.show()

# Menyimpan sinyal gelombang ke dalam format wav
wavfile.write("file1.wav", sr, signal2)


# Modifikasi kode (a) agar membuat gelombang suara dengan frekuensi 400
# Menentukan sampling rate
sr = 44100
# Menentukan frequency
freq = 400
# Menentukan length/panjang
length = 1
# Membuat fungsi linear dari 0 s/d length dengan titik berjumlah sr
t = np.arange(0, length, 1.0/sr)
# Membuat wave dengan sin function wave = A*sin(2*pi*freq*t)
signal3 = np.sin(np.pi*2*freq*t)
plt.plot(t, signal3)
plt.show()


# Modifikasi kode (b) dengan mengganti nilai amplitudo menjadi 50
# Menentukan sampling rate
sr = 44100
# Menentukan frequency dalam 1 detik
freq = 200
# Menentukan time (sumbu x)
length = 1

# Membuat fungsi linear dari 0 s/d time dengan titik berjumlah sr
# 1/sr --> stepsize, karena kita mau ada sr titik per detik
t = np.arange(0, length, 1.0/sr)

# Membuat wave dengan sin function wave = A*sin(2*pi*freq*t)
amplitude = 50
signal4 = amplitude * np.sin(np.pi*2*freq*t)

plt.plot(t, signal4)
plt.title('Gelombang Sinusoidal dengan Amplitudo 50')
plt.show()


# Import audio
audio_file = "C:/Users/FARIZA SHIELDA/Documents/File Unair/Semester 5/Data Mining II/Week 3/Test.wav"
audio_data, sr = librosa.load(audio_file)

# Visualisasi audio
plt.figure(figsize=(10, 4))
plt.plot(audio_data, color='b') 
plt.xlabel('Waktu (detik)')
plt.ylabel('Amplitudo')
plt.title('Visualisasi Audio')
plt.show()

# Mengubah sampling rate suara yang sama menjadi jauh lebih rendah
# Import audio dengan sampling rate asli
audio_file = "C:/Users/FARIZA SHIELDA/Documents/File Unair/Semester 5/Data Mining II/Week 3/Test.wav"

# Membaca data audio dari file dengan sampling rate 2300 Hz
data, sr = librosa.load(audio_file, sr=2300)

# Visualisasi
plt.figure(figsize=(10, 4))
plt.plot(audio_data, color='b')
plt.xlabel("Waktu (detik)")
plt.ylabel("Amplitudo")
plt.title("Visualisasi sampling rate yang telah direndahkan")
plt.show()

# Menyimpan data audio ke dalam file .wav dengan sampling rate 2300 Hz
output_file = "Test_rendah.wav"
wavfile.write(output_file, sr, data)
print(f"Data audio disimpan dalam file {output_file}")