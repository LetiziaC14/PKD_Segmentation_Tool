FloodFill interactive segmentation
================================

Prerequisiti
- Python 3.8+
- OpenCV and numpy

Installazione
```
pip install opencv-python numpy
```

Uso
```
python floodfill_segmentation.py -i /path/to/images -o masks -t 10
```

Controlli
- Left-click: aggiungi seed e crea una maschera usando `cv2.floodFill`.
- Trackbar `tol`: regola la tolleranza (loDiff/upDiff).
- `s`: salva le maschere create per l'immagine corrente nella cartella di output.
- `r`: reset delle maschere per l'immagine corrente.
- `u`: annulla l'ultima maschera creata (undo).
- Trackbar `radius`: raggio spaziale (in pixel) attorno al seed; la maschera sarà limitata ai pixel entro questo raggio e con intensità simile.
- `n`: immagine successiva.

- `m`: toggle modalità manuale (paint). In modalità manuale puoi disegnare direttamente sulle maschere:
	- sinistro: dipingi (aggiungi) pixel alla maschera corrente
	- destro: cancella (erase) pixel dalla maschera corrente
	- `brush` trackbar: dimensione del pennello in pixel
	- `c`: crea una nuova maschera vuota da editare
- `p`: immagine precedente.
- `q`: esci.

Output
- Le maschere vengono salvate in formato PNG binario (0/255) in `INPUT_FOLDER/masks` per default, come `image_basename_mask_1.png`, `image_basename_mask_2.png`, ecc.

- L'anteprima a video mostra l'immagine originale e a fianco un overlay con le maschere colorate (maschere multiple hanno colori distinti).
- Quando si preme `s` per salvare, lo script salva anche un file composito a colori (`image_basename_masks_overlay.png`) nella stessa cartella di output.
 - Quando si preme `s` per salvare, lo script salva:
	 - i singoli file binari per ogni maschera (`image_basename_mask_1.png`, ...),
	 - un file composito overlay (`image_basename_masks_overlay.png`),
	 - un unico file `image_basename_classmask.png` dove ogni classe (maschera) è colorata con un colore distinto (PNG RGB).
  
- Se non vuoi i singoli file binari, avvia lo script con `--no-binaries` per salvare solo il composito e il file class-colored:
	```
	python floodfill_segmentation.py -i path/to/images -o masks -t 10 --no-binaries
	```
  
- Premi `v` per alternare la visualizzazione: `both` (original + overlay), `overlay` (solo overlay colorato), `original` (solo immagine originale). Questo permette di lavorare su un'immagine alla volta.

Note
- Lo script non usa Docker né Label Studio; è standalone e lavora su immagini locali.
