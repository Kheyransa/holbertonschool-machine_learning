## Generative Adversarial Networks (GAN) — `unsupervised_learning/gan`

### Simple GAN — `0-simple_gan.py`

`class Simple_GAN(keras.Model):`

Sadə (vanilla) Generative Adversarial Network implementasiyası.

**Konstruktor:**
`def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):`

- `generator`: generator şəbəkəsi
- `discriminator`: discriminator şəbəkəsi
- `latent_generator`: latent vektorlar generasiya edən funksiya
- `real_examples`: real nümunələr tensoru
- `batch_size`: hər batch-in ölçüsü
- `disc_iter`: hər generator addımına düşən discriminator təlim
  addımlarının sayı
- `learning_rate`: hər iki optimizator üçün öyrənmə sürəti

**Loss funksiyaları:**
- Generator loss: `discriminator(generator(latent))` ilə `1` arasında MSE
- Discriminator loss: real nümunələr üçün `1`, fake nümunələr üçün `-1`
  hədəfləri ilə MSE-lərin cəmi

**Metodlar:**
- `get_real_sample(size=None)`: real nümunələrdən təsadüfi altçoxluq qaytarır
- `get_fake_sample(size=None, training=False)`: generator vasitəsilə
  fake nümunə yaradır
- `train_step(useless_argument)`: bir təlim addımı — `disc_iter` dəfə
  discriminator gradient descent, sonra bir dəfə generator gradient
  descent tətbiq edir və `{"discr_loss": ..., "gen_loss": ...}` qaytarır

### İstifadə

```python
Simple_GAN = __import__('0-simple_gan').Simple_GAN

generator, discriminator, latent_generator = fully_connected_GenDiscr(
    [1, 10, 10, 2], real_examples)
G = Simple_GAN(generator, discriminator, latent_generator, real_examples,
               learning_rate=.001)
G.compile()
G.fit(real_examples, epochs=16, steps_per_epoch=100)
```
