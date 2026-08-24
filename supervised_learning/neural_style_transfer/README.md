# Neural Style Transfer

Bu layihə TensorFlow istifadə edərək Neural Style Transfer (NST) həyata keçirir.

NST, əvvəlcədən öyrədilmiş VGG19 konvolyusiya şəbəkəsindən istifadə edərək bir

şəklin məzmununu (content) başqa bir şəklin üslubu (style — toxumalar, rənglər,

naxışlar) ilə birləşdirən yeni bir şəkil yaradır. Metod Gatys və b. tərəfindən

"A Neural Algorithm of Artistic Style" məqaləsinə əsaslanır.

## Qovluq

`supervised_learning/neural_style_transfer`

## Tələblər

- Python 3.x

- numpy

- tensorflow

- matplotlib (nümunələrdə şəkilləri göstərmək üçün)

## Fayllar

| Fayl | Təsvir |

| --- | --- |

| `0-neural_style.py` | `NST` sinfi: initialization və şəkil miqyaslama (`scale_image`) |

## Task 0: Initialize

`NST` sinfini müəyyən edir:

- Sinif atributları `style_layers` (style cost hesablamaq üçün istifadə

  olunan 5 VGG19 layı) və `content_layer` (content cost hesablamaq üçün

  istifadə olunan VGG19 layı).

- `__init__(self, style_image, content_image, alpha=1e4, beta=1)`: style və

  content şəkillərini yoxlayır, miqyaslayır və yaddaşda saxlayır, eləcə də

  `alpha` və `beta` çəkilərini saxlayır.

- `scale_image(image)` (static method): daxil edilən şəkli elə miqyaslayır ki,

  ən böyük tərəfi 512 piksel olsun (bicubic interpolasiya ilə) və piksel

  dəyərləri `[0, 1]` aralığına normallaşdırılsın. Nəticə `(1, h_new, w_new, 3)`

  formalı `tf.Tensor` olaraq qaytarılır.

### İstifadə nümunəsi

\`\`\`python

NST = __import__('0-neural_style').NST

nst = NST(style_image, content_image)

print(nst.style_image.shape)    # (1, h_new, w_new, 3)

print(nst.content_image.shape)  # (1, h_new, w_new, 3)

\`\`\`

## Müəllif

Holberton School - Machine Learning Specialization

Fatmaxanım
