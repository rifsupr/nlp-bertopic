# BERTopic - Advanced Topic Modeling

**Natural Language Processing (NLP)**

## Penjelasan Project

Project ini mengimplementasikan **BERTopic**, sebuah teknik *Topic Modeling* modern yang memanfaatkan *transformer embeddings* (BERT) dan *class-based TF-IDF* (c-TF-IDF) untuk menghasilkan topik yang padat dan mudah diinterpretasikan.

Berbeda dengan LDA tradisional, BERTopic menggunakan representasi semantik dari kata dan kalimat, sehingga mampu menangkap konteks yang lebih dalam. Project ini dikonfigurasi secara khusus untuk menangani dataset berukuran kecil dengan hasil yang optimal.

## Fitur Utama

-   **Contextual Embeddings**: Menggunakan model `all-MiniLM-L6-v2` dari Sentence-Transformers untuk mengubah teks menjadi vektor numerik yang kaya makna.
-   **Custom Vectorizer**: Menggunakan `CountVectorizer` dengan dukungan n-gram (1,2) untuk menangkap frasa kata majemuk.
-   **Dynamic Topic Reduction**: Kemampuan untuk mengelompokkan dokumen ke dalam topik secara otomatis berdasarkan kepadatan (*density-based clustering*).
-   **Interactive Visualization**: Menghasilkan visualisasi interaktif dalam format HTML untuk mengeksplorasi hubungan antar topik.

## Library yang Diinstall

Project ini menggunakan library terbaru dalam ekosistem NLP Python:

-   **bertopic**: Library utama untuk pemodelan topik.
-   **sentence-transformers**: Untuk menghasilkan embedding dari model BERT.
-   **pandas**: Untuk pengolahan dataset CSV.
-   **scikit-learn**: Untuk pra-pemrosesan teks (CountVectorizer).

Perintah instalasi:
```bash
pip install bertopic sentence-transformers pandas scikit-learn
```

## Penjelasan Dataset

Dataset yang digunakan adalah **`bbc-dataset.csv`** yang berisi ringkasan berita dari BBC News.
-   **Kolom**: `title` dan `description`.
-   **Pengolahan**: Kedua kolom digabungkan menjadi satu teks utuh untuk memberikan informasi yang lebih lengkap kepada model.

## Cara Menjalankan

1.  Pastikan Anda telah menginstall semua library yang dibutuhkan.
2.  **Penting**: Pastikan folder `output/` sudah tersedia di direktori project untuk menyimpan hasil visualisasi. Jika belum ada, buat manual:
    ```bash
    mkdir output
    ```
3.  Jalankan script utama:
    ```bash
    python main.py
    ```
4.  Hasil analisis topik akan muncul di terminal, dan file visualisasi akan tersimpan di folder `output/`.

## Hasil Output

Setelah dijalankan, project akan menghasilkan dua file visualisasi interaktif:
1.  **`topics.html`**: Visualisasi *Intertopic Distance Map* untuk melihat kedekatan antar topik dalam ruang 2D.
2.  **`barchart.html`**: Grafik batang yang menunjukkan kata kunci paling berpengaruh untuk setiap topik berdasarkan skor c-TF-IDF.

## Struktur Project

```text
bertopic/
├── datasets/
│   └── bbc-dataset.csv    # Dataset input berita BBC
├── output/                # Folder hasil visualisasi (HTML)
├── main.py                # Script utama implementasi BERTopic
└── README.md              # Dokumentasi project
```
