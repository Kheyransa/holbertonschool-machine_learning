# Autoencoders

Bu layihə Holberton School Machine Learning kursunun
`unsupervised_learning/autoencoders` bölməsinə aiddir.

## Fayllar

- `0-vanilla.py` — Sadə (vanilla) autoencoder yaradan `autoencoder` funksiyası.

## Tələblər

- Python 3.5+
- TensorFlow 2.x
- NumPy

## İstifadə

```python
autoencoder = __import__('0-vanilla').autoencoder

encoder, decoder, auto = autoencoder(784, [128, 64], 32)
auto.fit(x_train, x_train, epochs=50, batch_size=256,
         shuffle=True, validation_data=(x_test, x_test))
```

## Funksiya təsviri

`def autoencoder(input_dims, hidden_layers, latent_dims):`

- `input_dims`: modelin giriş ölçüsü (integer)
- `hidden_layers`: encoder-dəki gizli qatların node sayının siyahısı
  (decoder üçün tərsinə çevrilir)
- `latent_dims`: latent fəzanın ölçüsü (integer)

Qaytarır: `encoder, decoder, auto`

Autoencoder `adam` optimizatoru və `binary_crossentropy` itki funksiyası
ilə compile edilir. Bütün qatlarda `relu` aktivasiyası istifadə olunur,
decoderin son qatı isə `sigmoid` istifadə edir.
