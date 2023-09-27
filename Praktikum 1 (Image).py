# Import package
import cv2
import numpy as np

# Membaca citra
img = cv2.imread("C:/Users/FARIZA SHIELDA/Documents/File Unair/Semester 5/Data Mining II/Week 3/kitty.png")
cv2.imshow("Citra Kitty", img)
cv2.waitKey(0)

# Mengetahui ukuran dan matriks dari citra
print("Ukuran Citra Warna: ", img.shape)
print("Matriks dari Citra Warna pada baris 0 dan kolom 0: ", img[0,0])

# Memisahkan ketiga channel
(blue,green,red)=cv2.split(img)

# Menampilkan channel biru
cv2.imshow("Komponen Biru", blue)

# Membuat matrix berisi 0 yang berukuran sesuai image asli
zeroMatrix = np.zeros(img.shape[:2], img.dtype)
m = zeroMatrix
blue_img = cv2.merge([blue, m,m])
combine_img = np.hstack((img, blue_img))

# Menampilkan citra gabungan dalam satu jendela
cv2.imshow("Semua Citra", combine_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_img)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mengetahui ukuran dan matriks dari citra grayscale
print("Ukuran Citra Warna Grayscale: ", gray_img.shape)
print("Matriks dari Citra Warna Grayscale pada baris 0 dan kolom 0: ", gray_img[0,0])

# Define the coordinates of the region you want to crop
# Format: (y_start:y_end, x_start:x_end)
x_start, x_end = 0, 200  # Koordinat horizontal
y_start, y_end = 0, 170  # Koordinat vertikal

# Crop the image
cropped_img = img[y_start:y_end, x_start:x_end]

# Display the cropped image
cv2.imshow('Cropped Image', cropped_img)

# Wait for a key press and then close the window
cv2.waitKey(0)

# Menampilkan channel Green dan Red
zeroMatrix = np.zeros(img.shape[:2], img.dtype)
m = zeroMatrix
red = cv2.merge([m, m,red])
green = cv2.merge([m, green,m])

combined_img = np.hstack((img, green, red))

# Menampilkan citra gabungan dalam satu jendela
cv2.imshow("Semua Citra", combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Menyimpan citra ke dalam format JPG
cv2.imwrite("kitty.jpg", img)

# Menutup semua jendela
cv2.destroyAllWindows()

# Menampilkan array citra asli dan citra format terbaru
import cv2
img = cv2.imread("C:/Users/FARIZA SHIELDA/Documents/File Unair/Semester 5/Data Mining II/Week 3/kitty.png")
img2 = cv2.imread("C:/Users/FARIZA SHIELDA/Documents/File Unair/Semester 5/Data Mining II/Week 3/kitty.jpg")
img
img2