# Watermark-it


## Fonctionnalités


## Fonctionnement

YCbCr -> DWT LL -> 8x8 -> DCT -> Flipping -> IDCT -> IDWT LL -> YCbCr


### Méthode d'insertion

La méthode utilisée est celle du basculement de fréquence (Frequency-Flipping). Appliqué aux coefficients obtenus par la DCT des blocs de 8x8 pixel de l'image (matrice de normalisation DCT). L'idée est d'inverser 2 des coefficients de chaque bloc s'il ne respecte pas la règle suivante :

C1 >  C2 --> 0
C1 <= C2 --> 1

(avec C1 et C2 les coefficients à comparer et 0 et 1 les valeurs à attribuer correspondant à la marque en binaire)

L'emplacement des deux coefficients DCT devraient avoir une valeur moyenne comparable pour la matrice de normalisation DCT 8x8. (ex : (4, 1) et (2, 3))


## Exemple d'utilisation

### Watermarking avec une image
Fichier `Watermark.py`
```python
watermarkeImage = embeddedImage(image,marque, password) 
watermarkeImage.save(result)

watermarkArray = recoverWatermark(result, password) # Wsize=(x, y) for specific mark
watermarkArray.save('result/recoveredWatermark.png')
```

### Affichage des images et comparaison
Fichier `Plot.py`
```python
plotResult(image, marque, Iresult, Mresult, x)

plotDiff(image, marque, Iresult, Mresult, x) 
```
### Attaque de l'image watermarkée
Fichier `Attack.py`
```python
# Vue groupée
attackAll(image, marque, Iresult, Mresult, x, password)
# Vue individuelle
attackOne(image, marque, Iresult, Mresult, x, password)
```

### Watermarking avec du texte
```python
watermarkeImage = embeddedTexte(image,texte, password)
watermarkeImage.save(result)

watermarkArray = recoverText(result, password)
print(watermarkArray)
```

## Application

Intégration dans une application Android : [Watermarker](https://github.com/Skelrin/Watermarker)


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
