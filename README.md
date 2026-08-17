# Katalóg vzdelávacích materiálov

Streamlit prototyp na vyhľadávanie materiálov zo súboru `zs_vycistne.csv` a ich
prepojenie na vzdelávacie štandardy v `SVPv5.csv`.

## Funkcie

- fulltextové vyhľadávanie v názve (`resource_title`) a popise (`description`),
- filtrovanie v hornej lište podľa predmetu a cyklu,
- voliteľné zúženie výsledkov podľa vzdelávacieho štandardu,
- zobrazenie všetkých zodpovedajúcich materiálov, kým nie je vybraný štandard,
- stránkovanie výsledkov a priame odkazy na materiály vo Viki,
- vyhľadávanie bez ohľadu na veľkosť písmen a diakritiku.
- vizuálny štýl prispôsobený portálu Viki (žlté akcie a karty materiálov).

Aplikácia akceptuje názov `zs_vycistene.csv` zo zadania aj aktuálny názov súboru
`zs_vycistne.csv`.

## Spustenie

```bash
pip install -r requirements.txt
streamlit run app.py
```

CSV súbory musia byť uložené v rovnakom priečinku ako `app.py`.

## Optimalizácia dát

Aplikácia načítava komprimované súbory `data/materials.parquet` a
`data/standards.parquet`. Obsahujú iba stĺpce potrebné pre aplikáciu, zjednotené
štandardy a predpočítaný text na vyhľadávanie. Pre samotný beh aplikácie preto
pôvodné CSV nie sú potrebné; použijú sa iba ako fallback, ak optimalizovaný
balík chýba alebo je poškodený.

Po zmene niektorého zdrojového CSV obnovte optimalizované dáta príkazom:

```bash
python optimize_data.py
```

Manifest obsahuje kontrolné súčty zdrojov pre audit a údaje na kontrolu
integrity optimalizovaného balíka.
