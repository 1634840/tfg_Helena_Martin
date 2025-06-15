# TFG Helena Martin

Aquest repositori conté el codi i les dades utilitzades per al treball de fi de grau, enfocat en l'anàlisi de punts de pressió, articulacions i xarxes neuronals aplicades a dades biomèdiques.

## 📁 Contingut

- `comparar_xarxes.py`: Script per comparar el rendiment diferents xarxes neuronals per la deetccio de les articulacions.
- `dataset_uab_etiquetat.jsonl`: Fitxer amb el conjunt de dades etiquetat (format JSON Lines).
- `model_heatmap.py`: Model que prediu la postura a partir de mapes de pressió.
- `model_heatmap_articulacions.py`:  Model amb l'arquitectura de xarxa DenseNet i ResNet que prediu la postura a partir de mapes de pressió i el vector d'articulacions pressionades.
- `punts_pressio.py`: Model que prediu primer la posicio de les articulacions a l'esquelet i posteriorment els punts de pressió.
