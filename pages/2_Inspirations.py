from pathlib import Path
import math
import re
import unicodedata
import json

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
IDEES_VACANCES_FILE = BASE_DIR / "idees_vacances.json"
CSV_FILE = BASE_DIR / "places.csv"
CITIES_FILE = BASE_DIR / "cities15000.txt"
COUNTRIES_FILE = BASE_DIR / "countryInfo.txt"

RADIUS_KM = 30
EARTH_RADIUS_KM = 6371.0088


st.title("Inspirations de voyage")
st.write(
    "Choisis un pays et une ville pour découvrir les lieux remarquables "
    "situés à proximité."
)


# --------------------------------------------------
# Normalisation des noms
# --------------------------------------------------

def normalize_name(value):
    """Normalise un nom pour faciliter les comparaisons."""
    if pd.isna(value):
        return ""

    value = str(value).strip().casefold()
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", value)


# --------------------------------------------------
# Chargement des pays
# --------------------------------------------------

@st.cache_data
def load_countries():
    """Charge les codes ISO et les noms depuis countryInfo.txt."""
    countries = {}

    if not COUNTRIES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {COUNTRIES_FILE}"
        )

    with open(
        COUNTRIES_FILE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue

            fields = line.rstrip("\n\r").split("\t")

            if len(fields) < 5:
                continue

            iso_code = fields[0].strip().upper()
            country_name = fields[4].strip()

            if (
                len(iso_code) == 2
                and country_name
                and iso_code.isalpha()
            ):
                countries[iso_code] = country_name

    return countries


# --------------------------------------------------
# Codes pays présents dans places.csv
# --------------------------------------------------

@st.cache_data
def load_place_country_codes():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CSV_FILE}"
        )

    places = pd.read_csv(
        CSV_FILE,
        usecols=["iso"],
        dtype={"iso": "string"}
    )

    codes = (
        places["iso"]
        .dropna()
        .str.strip()
        .str.upper()
    )

    return {
        code for code in codes
        if len(code) == 2 and code.isalpha()
    }


# --------------------------------------------------
# Chargement des villes
# --------------------------------------------------

@st.cache_data
def load_cities():
    if not CITIES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CITIES_FILE}"
        )

    columns = [
        "geonameid",
        "name",
        "asciiname",
        "alternatenames",
        "latitude",
        "longitude",
        "feature_class",
        "feature_code",
        "country_code",
        "cc2",
        "admin1_code",
        "admin2_code",
        "admin3_code",
        "admin4_code",
        "population",
        "elevation",
        "dem",
        "timezone",
        "modification_date",
    ]

    cities = pd.read_csv(
        CITIES_FILE,
        sep="\t",
        header=None,
        names=columns,
        usecols=[
            "geonameid",
            "name",
            "latitude",
            "longitude",
            "country_code",
            "population",
            "feature_class",
            "feature_code",
        ],
        dtype={
            "geonameid": "Int64",
            "name": "string",
            "country_code": "string",
            "feature_class": "string",
            "feature_code": "string",
        },
        low_memory=False,
    )

    cities["latitude"] = pd.to_numeric(
        cities["latitude"], errors="coerce"
    )
    cities["longitude"] = pd.to_numeric(
        cities["longitude"], errors="coerce"
    )
    cities["population"] = pd.to_numeric(
        cities["population"], errors="coerce"
    )

    cities = cities.dropna(
        subset=[
            "geonameid",
            "name",
            "latitude",
            "longitude",
            "country_code",
        ]
    ).copy()

    cities["country_code"] = (
        cities["country_code"].str.strip().str.upper()
    )

    cities = cities[
        cities["latitude"].between(-90, 90)
        & cities["longitude"].between(-180, 180)
        & cities["name"].str.strip().ne("")
    ].copy()

    # Exclure les arrondissements pour éviter les doublons
    cities = cities[
        ~cities["name"].str.contains(
            r"arrondissement",
            case=False,
            na=False
        )
    ].copy()

    # Regrouper les variantes d'un même nom dans un pays
    cities["_normalized_name"] = cities["name"].map(
        normalize_name
    )

    cities["population"] = cities["population"].fillna(0)

    cities = cities.sort_values(
        "population",
        ascending=False
    )

    cities = cities.drop_duplicates(
        subset=["country_code", "_normalized_name"],
        keep="first",
    )

    return cities


# --------------------------------------------------
# Chargement des données
# --------------------------------------------------

try:
    countries = load_countries()
    place_codes = load_place_country_codes()
    cities = load_cities()

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Impossible de charger les données : {error}")
    st.stop()


# --------------------------------------------------
# Sélection du pays
# --------------------------------------------------

city_codes = set(cities["country_code"].dropna())

available_codes = (
    set(countries.keys())
    & place_codes
    & city_codes
)

if not available_codes:
    st.error(
        "Aucun pays commun aux fichiers de pays, de villes "
        "et de lieux n'a été trouvé."
    )
    st.stop()


country_counts = {
    code: sum(
        1 for other_code in available_codes
        if normalize_name(countries[other_code])
        == normalize_name(countries[code])
    )
    for code in available_codes
}

country_labels = {
    code: (
        f"{countries[code]} ({code})"
        if country_counts[code] > 1
        else countries[code]
    )
    for code in available_codes
}

country_options = sorted(
    available_codes,
    key=lambda code: normalize_name(countries[code])
)

default_country_index = (
    country_options.index("FR")
    if "FR" in country_options
    else 0
)

selected_country = st.selectbox(
    "Pays",
    options=country_options,
    index=default_country_index,
    format_func=lambda code: country_labels[code],
)


# --------------------------------------------------
# Sélection de la ville
# --------------------------------------------------

country_cities = cities[
    cities["country_code"] == selected_country
].copy()

country_cities = country_cities.sort_values(
    ["population", "name"],
    ascending=[False, True],
)

if country_cities.empty:
    st.warning("Aucune ville disponible pour ce pays.")
    st.stop()

city_lookup = country_cities.set_index(
    "geonameid",
    drop=False
).to_dict("index")

city_ids = country_cities["geonameid"].tolist()

selected_city_id = st.selectbox(
    "Ville de référence",
    options=city_ids,
    format_func=lambda city_id: city_lookup[city_id]["name"],
)

selected_city = city_lookup[selected_city_id]

city_name = selected_city["name"]
city_latitude = float(selected_city["latitude"])
city_longitude = float(selected_city["longitude"])


# --------------------------------------------------
# Gestion des idées de vacances
# --------------------------------------------------

def load_idees_vacances():
    """Charge les idées de vacances depuis le fichier JSON."""
    if not IDEES_VACANCES_FILE.exists():
        return []

    try:
        with open(
            IDEES_VACANCES_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    # Compatibilité avec les anciens fichiers contenant
    # uniquement une liste de noms de villes.
    idees_vacances = []

    for item in data:
        if isinstance(item, str):
            idees_vacances.append({
                "city": item,
                "country": "",
            })

        elif isinstance(item, dict) and item.get("city"):
            idees_vacances.append({
                "city": str(item["city"]),
                "country": str(item.get("country", "")),
            })

    return idees_vacances


def save_idees_vacances(idees_vacances):
    """Enregistre les idées de vacances dans le fichier JSON."""
    with open(
        IDEES_VACANCES_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            idees_vacances,
            file,
            ensure_ascii=False,
            indent=2
        )


idees_vacances = load_idees_vacances()

already_saved = any(
    item["city"] == city_name
    and item["country"] == selected_country
    for item in idees_vacances
)

if already_saved:
    st.success(
        f"⭐ {city_name}, {countries[selected_country]} "
        "est déjà dans tes idées de vacances."
    )
else:
    if st.button("⭐ Ajouter à mes idées de vacances"):
        idees_vacances.append({
            "city": city_name,
            "country": selected_country,
        })

        save_idees_vacances(idees_vacances)
        st.success(f"{city_name} a été ajoutée à tes idées de vacances !")
        st.rerun()


with st.expander(
    f"Mes idées de vacances ({len(idees_vacances)})"
):
    if idees_vacances:
        for index, item in enumerate(idees_vacances):
            city = item["city"]
            country_code = item.get("country", "")
            country_name = countries.get(
                country_code,
                country_code
            )

            col_city, col_delete = st.columns([4, 1])

            with col_city:
                if country_name:
                    st.write(f"⭐ {city}, {country_name}")
                else:
                    st.write(f"⭐ {city} (pays à préciser)")

            with col_delete:
                if st.button(
                    "🗑️",
                    key=f"delete_idee_vacances_{index}"
                ):
                    idees_vacances.pop(index)
                    save_idees_vacances(idees_vacances)
                    st.rerun()
    else:
        st.write("Aucune idée de vacances enregistrée pour le moment.")


# --------------------------------------------------
# Résumé de la sélection
# --------------------------------------------------

st.divider()
st.subheader("Ta sélection")

st.write(f"**Pays :** {country_labels[selected_country]}")
st.write(f"**Ville :** {city_name}")
st.write(
    f"**Coordonnées :** "
    f"{city_latitude:.5f}, {city_longitude:.5f}"
)

st.caption(
    f"Les lieux situés dans un rayon de {RADIUS_KM} km "
    "seront affichés, toutes frontières confondues."
)


# --------------------------------------------------
# Chargement des lieux
# --------------------------------------------------

@st.cache_data
def load_places():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CSV_FILE}"
        )

    places = pd.read_csv(CSV_FILE, low_memory=False)

    required_columns = {
        "name",
        "latitude",
        "longitude",
    }

    missing_columns = required_columns - set(places.columns)

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans places.csv : "
            + ", ".join(sorted(missing_columns))
        )

    places["latitude"] = pd.to_numeric(
        places["latitude"], errors="coerce"
    )

    places["longitude"] = pd.to_numeric(
        places["longitude"], errors="coerce"
    )

    places = places.dropna(
        subset=["name", "latitude", "longitude"]
    ).copy()

    places = places[
        places["latitude"].between(-90, 90)
        & places["longitude"].between(-180, 180)
    ].copy()

    # Convertir les colonnes de classement en valeurs numériques
    for column in ["score", "fame_rank", "kind_rank"]:
        if column in places.columns:
            places[column] = pd.to_numeric(
                places[column], errors="coerce"
            )

    return places


# --------------------------------------------------
# Calcul des distances
# --------------------------------------------------

def distances_from_point(df, latitude, longitude):
    """Calcule les distances entre un point et tous les lieux."""

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)

    lat2 = df["latitude"].map(math.radians)
    lon2 = df["longitude"].map(math.radians)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        delta_lat.map(
            lambda value: math.sin(value / 2) ** 2
        )
        + math.cos(lat1)
        * lat2.map(math.cos)
        * delta_lon.map(
            lambda value: math.sin(value / 2) ** 2
        )
    )

    a = a.clip(0, 1)

    return a.map(
        lambda value: 2
        * EARTH_RADIUS_KM
        * math.asin(math.sqrt(value))
    )


# --------------------------------------------------
# Recherche des lieux à proximité
# --------------------------------------------------

try:
    places = load_places()

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Impossible de charger les lieux : {error}")
    st.stop()


places["distance_km"] = distances_from_point(
    places,
    city_latitude,
    city_longitude,
)

nearby = places[
    places["distance_km"] <= RADIUS_KM
].copy()


# --------------------------------------------------
# Classement des lieux
# --------------------------------------------------

if "score" in nearby.columns and nearby["score"].notna().any():

    # Score numérique : plus élevé en premier
    nearby = nearby.sort_values(
        ["score", "distance_km"],
        ascending=[False, True],
        na_position="last",
    )

elif "fame_rank" in nearby.columns:

    # Un rang de célébrité plus petit est meilleur
    nearby = nearby.sort_values(
        ["fame_rank", "distance_km"],
        ascending=[True, True],
        na_position="last",
    )

else:
    # À défaut de score ou de rang, lieux les plus proches
    nearby = nearby.sort_values(
        "distance_km",
        ascending=True,
    )


# --------------------------------------------------
# Affichage des résultats
# --------------------------------------------------

st.divider()
st.subheader("Lieux remarquables à proximité")

st.write(
    f"**{len(nearby)} lieu(x) trouvé(s)** "
    f"dans un rayon de {RADIUS_KM} km."
)

if nearby.empty:
    st.info(
        "Aucun lieu n'a été trouvé dans ce rayon. "
        "Essaie une autre ville."
    )
    st.stop()


for position, (_, place) in enumerate(
    nearby.iterrows(),
    start=1
):

    with st.container(border=True):

        col_info, col_image = st.columns(
            [2, 1],
            gap="medium",
        )

        with col_info:
            st.markdown(f"### {position}. {place['name']}")

            st.caption(
                f"Distance : {place['distance_km']:.1f} km"
            )

            if (
                "score" in nearby.columns
                and pd.notna(place.get("score"))
            ):
                st.write(f"**Score :** {place['score']}")

            elif (
                "fame_rank" in nearby.columns
                and pd.notna(place.get("fame_rank"))
            ):
                st.write(
                    f"**Rang de célébrité :** "
                    f"{int(place['fame_rank'])}"
                )

            kind_label = place.get("kind_label")

            if (
                pd.notna(kind_label)
                and str(kind_label).strip()
            ):
                st.write(f"**Type :** {kind_label}")

            description = place.get("description")

            if (
                pd.notna(description)
                and str(description).strip()
            ):
                st.write(str(description).strip())

            wikipedia_url = place.get("wikipedia")
            map_url = place.get("map_url")

            links = []

            if (
                pd.notna(wikipedia_url)
                and str(wikipedia_url).startswith("https://")
            ):
                links.append(
                    f"[Lire sur Wikipédia]({wikipedia_url})"
                )

            if (
                pd.notna(map_url)
                and str(map_url).startswith("https://")
            ):
                links.append(
                    f"[Voir sur la carte]({map_url})"
                )

            if links:
                st.markdown(" · ".join(links))

        with col_image:
            image_url = place.get("image")

            if (
                pd.notna(image_url)
                and str(image_url).strip().startswith("https://")
            ):
                st.image(
                    str(image_url).strip(),
                    width=280,
                )
            else:
                st.caption("Aucune photo disponible.")