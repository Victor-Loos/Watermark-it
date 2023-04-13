# Watermark-it


## Fonctionnalités

### Point fort
- Imperceptible (PSNR >= 35dB)
- Couleur de l'image principale conservée
- Taille de l'image principale conservée (à 7x7 pixels près)
- Placement de la marque pseudo-aléatoire (avec un mot de passe)
- Extraction aveugle
- Robustesse aux attaques (compression, bruit, redimensionnement ± 2)
- Conversion automatique des images (taille, couleur, format binaire)
- Taille de la marque ajustée automatiquement ((1/16)/1,3 de l'image)
- Marque format texte ou image

### Limites
- Format d'image compatible : PNG, JPEG, BMP
- Faible espace de stockage (1/16 de l'image max)
- Non robuste aux attaques (rotation, déformation)



## Fonctionnement

YCbCr -> DWT LL -> 8x8 -> DCT -> Flipping -> IDCT -> IDWT LL -> YCbCr

(1/16)/1,3 de l'image : Pour calculer plus rapidement les positions
Conversion de l'image au multiple de 8 inférieurs le plus proche : Pour l'utilisation de la DCT sur les blocs de 8x8 pixels


### Méthode d'insertion

La méthode utilisée est celle du basculement de fréquence (Frequency-Flipping). Appliqué aux coefficients obtenus par la DCT des blocs de 8x8 pixel de l'image (matrice de normalisation DCT). L'idée est d'inverser 2 des coefficients de chaque bloc s'il ne respecte pas la règle suivante :

$$\begin{aligned}
& C1 >\quad C2 \longrightarrow 0 \\
& C1 <= C2 \longrightarrow 1
\end{aligned}$$

(avec C1 et C2 les coefficients à comparer et 0 et 1 les valeurs à attribuer correspondant à la marque en binaire)

L'emplacement des deux coefficients DCT devraient avoir une valeur moyenne comparable pour la matrice de normalisation DCT 8x8. (ex : (4, 1) et (2, 3))


## Exemple d'utilisation

### Watermarking avec une image
Fichier `watermark.py`
```python
watermarkeImage = embeddedImage(image,marque, password) 
watermarkeImage.save(result)

watermarkArray = recoverWatermark(result, password) # Wsize=(x, y) for specific mark
watermarkArray.save('result/recoveredWatermark.png')
```

#### Affichage des images et comparaison
Fichier `display.py`
```python
plotResult(image, marque, Iresult, Mresult, x)

plotDiff(image, marque, Iresult, Mresult, x) 
```
#### Attaque de l'image watermarkée
Fichier `attack.py`
```python
# Vue groupée
attackAll(image, marque, Iresult, Mresult, x, password)
# Vue individuelle
attackOne(image, marque, Iresult, Mresult, x, password)
```

### Watermarking avec du texte
Fichier `watermark.py`
```python
watermarkeImage = embeddedTexte(image,texte, password)
watermarkeImage.save(result)

watermarkArray = recoverText(result, password)
print(watermarkArray)
```

## Résultats

Différences

![display1](./result/display1.png)
![display2](./result/display2.png)


Robustesse / Attaques

![attack](./result/attack.png)


## Application

Intégration dans l'application Android : [Watermarker](https://github.com/Skelrin/Watermarker)


### Archive

Ordre d'avancement des fichiers : 

Image Watermarking.ipynb
- Arnold
- Colors
- Texte
- YCbCr
- Key
- Flipping
- Optimisation
- Position
- Size


---

Référence : 
[Image-watermarking-using-DCT](https://github.com/voilentKiller0/Image-watermarking-using-DCT)

Intéréssant :
[blind_watermark](https://github.com/guofei9987/blind_watermark) (best ?)
[DWT-and-DCT-watermarking](https://github.com/ChanonTonmai/DWT-and-DCT-watermarking) (non robuste)
[DCT-DWT-SVD-Digital-Watermarking](https://github.com/cyanaryan/DCT-DWT-SVD-Digital-Watermarking) (non aveugle)
[image_watermarking](https://github.com/lakshitadodeja/image_watermarking) (ex for dwt, dct, dft,svd and dwt-svd)

