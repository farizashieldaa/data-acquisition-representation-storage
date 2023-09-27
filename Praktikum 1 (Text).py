char = 'A'
print(ord(char))

ascii = 65
print(chr(ascii))

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

docs = 'I ate an apple'

#memisah kalimat menjadi kolom
split_docs = docs.split(' ')
data = [doc.split(' ') for doc in split_docs]
values = array(data).ravel()

#integer code
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

#binary encode
onehot_encoder=OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

text = ["everybody love nlp", "nlp is so cool", 
"nlp is all about helping machines process language", 
"this tutorial is on basic nlp technique"]

vectorizer = CountVectorizer()

# tokenisasi dan membuat vocab
vectorizer.fit(text)
print(vectorizer.vocabulary_)

# encode dokumen
vector = vectorizer.transform(text)

# hasil encode vektor
print(vector.shape) 
print(vector.toarray())

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
text1 = ['i love nlp', "nlp is so cool", 
"nlp is all about helping machines process language", 
"this tutorial is on basic nlp technique"]

tf = TfidfVectorizer()
txt_fitted = tf.fit(text1)
txt_transformed = txt_fitted.transform(text1)

idf = tf.idf_
print(dict(zip(txt_fitted.get_feature_names_out(), idf)))

# Modifikasi kode (a) agar bisa menampilkan ASCII code untuk "datA mining"

text = 'datA mining'
for char in text:
    ascii_code = ord(char)
    print(f"Karakter '{char}' memiliki ASCII code: {ascii_code}")


# Tambahkan kode (b) agar bisa menampilkan kembali kata pertama yang dilakukan One-Hot Encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

docs = "I ate an apple"

# Memisahkan kalimat menjadi token
split_docs = docs.split(" ")
data = [doc.split(" ") for doc in split_docs]
values = np.array(data).ravel()

# Integer Encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# Binary Encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# Menampilkan kata pertama yang dilakukan one-hot encoding
first_word_encoded = onehot_encoded[0].reshape(1, -1)
# Mengembalikan representasi integer dari encoding one-hot
first_word_integer_encoded = np.argmax(first_word_encoded)
# Mengembalikan kata pertama dalam bentuk string sesuai dengan encoding one-hot
first_word_original = label_encoder.inverse_transform([first_word_integer_encoded])

print(f"Kata pertama dalam one-hot encoding: {first_word_encoded}")
print(f"Kata pertama dalam bentuk integer: {first_word_integer_encoded}")
print(f"Kata pertama dalam bentuk string: {first_word_original[0]}")


# Impor modul CountVectorizer dan TfidfVectorizer dari scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Membaca teks dari file "gutenberg.txt"
with open("gutenberg.txt", "r", encoding="utf-8") as file:
    corpus = file.read()

# Inisialisasi CountVectorizer dengan pengaturan untuk mengabaikan kata-kata stop (stop_words="english")
count_vectorizer = CountVectorizer(stop_words="english")

# Melakukan transformasi menggunakan CountVectorizer pada teks yang dibaca
count_vector = count_vectorizer.fit_transform([corpus])

# Mendapatkan daftar fitur (kata-kata) dari hasil CountVectorizer
count_feature_names = count_vectorizer.get_feature_names_out()

# Inisialisasi TF-IDF Vectorizer dengan pengaturan untuk mengabaikan kata-kata stop (stop_words="english")
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Melakukan transformasi menggunakan TF-IDF Vectorizer pada teks yang dibaca
tfidf_vector = tfidf_vectorizer.fit_transform([corpus])

# Mendapatkan daftar fitur (kata-kata) dari hasil TF-IDF Vectorizer
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Hasil CountVectorizer
print("Hasil CountVectorizer:")
print(count_feature_names)

# Hasil TF-IDF Vectorizer
print("\nHasil TF-IDF Vectorizer:")
print(tfidf_feature_names)


# Import library yang diperlukan
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Menampilkan judul
print("\nTF-IDF")

# Membuat daftar teks
text1 = ['i love nlp', "nlp is so cool", 
"nlp is all about helping machines process language", 
"this tutorial is on basic nlp technique"]

# Membuat objek TfidfVectorizer
tf = TfidfVectorizer()

# Menyesuaikan (fit) teks dengan objek TfidfVectorizer
txt_fitted = tf.fit(text1)

# Mengubah teks menjadi representasi TF-IDF
txt_transformed = txt_fitted.transform(text1)

# Menghitung nilai IDF untuk setiap kata
idf = dict(zip(txt_fitted.get_feature_names_out(), txt_fitted.idf_))

# Menampilkan grafik
plt.figure(figsize=(10, 6))
plt.bar(idf.keys(), idf.values())
plt.xlabel('Kata')
plt.ylabel('Nilai TF-IDF')
plt.title('TF-IDF untuk Setiap Kata')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
