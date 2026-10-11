from pathlib import Path
import json
import math
import re
import unicodedata

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
MAX_PLACES = 5


# --------------------------------------------------
# Interface
# --------------------------------------------------

st.title("Mes prochaines vacances")

st.write(
    "Retrouve les lieux remarquables à visiter autour "
    "des villes que tu as enregistrées."
)

st.caption(
    "Organisation par pays et par ville. "
    "Jusqu'à 5 lieux par ville, dans un rayon strict de 30 km."
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
# Chargement des idées de vacances
# --------------------------------------------------

def load_idees_vacances():
    """Charge les villes enregistrées dans le fichier JSON."""

    if not IDEES_VACANCES_FILE.exists():
        return []

    try:
        with open(
            IDEES_VACANCES_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (json.JSONDecodeError, OSError):
        st.error(
            "Impossible de lire idees_vacances.json."
        )
        return []

    if not isinstance(data, list):
        st.error(
            "Le fichier idees_vacances.json doit contenir une liste."
        )
        return []

    idees = []

    for item in data:
        # Compatibilité avec les anciens favoris enregistrés
        # sous forme de simples noms de villes.
        if isinstance(item, str):
            idees.append({
                "city": item.strip(),
                "country": "",
            })

        elif isinstance(item, dict) and item.get("city"):
            idees.append({
                "city": str(item["city"]).strip(),
                "country": str(
                    item.get("country", "")
                ).strip().upper(),
            })

    # Éliminer les doublons exacts.
    unique_idees = []
    seen = set()

    for idee in idees:
        key = (
            normalize_name(idee["city"]),
            idee["country"],
        )

        if key not in seen:
            seen.add(key)
            unique_idees.append(idee)

    return unique_idees


# --------------------------------------------------
# Chargement des pays
# --------------------------------------------------

@st.cache_data
def load_countries():
    """Charge les codes ISO et les noms des pays."""

    if not COUNTRIES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {COUNTRIES_FILE}"
        )

    countries = {}

    with open(
        COUNTRIES_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue

            fields = line.rstrip("\n\r").split("\t")

            if len(fields) < 5:
                continue

            code = fields[0].strip().upper()
            name = fields[4].strip()

            if (
                len(code) == 2
                and code.isalpha()
                and name
            ):
                countries[code] = name

    return countries


# --------------------------------------------------
# Chargement des villes
# --------------------------------------------------

@st.cache_data
def load_cities():
    """Charge les coordonnées et la population des villes."""

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
        ],
        dtype={
            "geonameid": "Int64",
            "name": "string",
            "country_code": "string",
        },
        low_memory=False,
    )

    for column in [
        "latitude",
        "longitude",
        "population",
    ]:
        cities[column] = pd.to_numeric(
            cities[column],
            errors="coerce",
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

    cities = cities[
        ~cities["name"].str.contains(
            "arrondissement",
            case=False,
            na=False,
        )
    ].copy()

    cities["_normalized_name"] = cities["name"].map(
        normalize_name
    )

    cities["population"] = cities["population"].fillna(0)

    cities = cities.sort_values(
        "population",
        ascending=False,
    )

    cities = cities.drop_duplicates(
        subset=["country_code", "_normalized_name"],
        keep="first",
    )

    return cities


# --------------------------------------------------
# Chargement des lieux remarquables
# --------------------------------------------------

@st.cache_data
def load_places():
    """Charge les lieux depuis places.csv."""

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CSV_FILE}"
        )

    places = pd.read_csv(
        CSV_FILE,
        low_memory=False,
    )

    required_columns = {
        "name",
        "latitude",
        "longitude",
    }

    missing = required_columns - set(places.columns)

    if missing:
        raise ValueError(
            "Colonnes manquantes dans places.csv : "
            + ", ".join(sorted(missing))
        )

    places["latitude"] = pd.to_numeric(
        places["latitude"],
        errors="coerce",
    )

    places["longitude"] = pd.to_numeric(
        places["longitude"],
        errors="coerce",
    )

    places = places.dropna(
        subset=["name", "latitude", "longitude"]
    ).copy()

    places = places[
        places["latitude"].between(-90, 90)
        & places["longitude"].between(-180, 180)
    ].copy()

    for column in [
        "score",
        "fame_rank",
        "kind_rank",
    ]:
        if column in places.columns:
            places[column] = pd.to_numeric(
                places[column],
                errors="coerce",
            )

    return places


# --------------------------------------------------
# Calcul des distances
# --------------------------------------------------

def distances_from_point(df, latitude, longitude):
    """Calcule les distances à vol d'oiseau en kilomètres."""

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
# Classement des lieux
# --------------------------------------------------

def rank_places(places):
    """Classe les lieux selon le score, le rang ou la distance."""

    if (
        "score" in places.columns
        and places["score"].notna().any()
    ):
        return places.sort_values(
            ["score", "distance_km"],
            ascending=[False, True],
            na_position="last",
        )

    if (
        "fame_rank" in places.columns
        and places["fame_rank"].notna().any()
    ):
        return places.sort_values(
            ["fame_rank", "distance_km"],
            ascending=[True, True],
            na_position="last",
        )

    return places.sort_values(
        "distance_km",
        ascending=True,
    )


# --------------------------------------------------
# Résolution des villes enregistrées
# --------------------------------------------------

try:
    countries = load_countries()
    cities = load_cities()
    places = load_places()

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(
        f"Impossible de charger les données : {error}"
    )
    st.stop()


idees_vacances = load_idees_vacances()

if not idees_vacances:
    st.info(
        "Tu n'as pas encore enregistré de ville. "
        "Rends-toi dans « Inspirations de voyage » "
        "pour ajouter tes premières idées."
    )
    st.stop()


# On résout les coordonnées avant de regrouper les villes
# par pays. Cela permet aussi de gérer les anciens favoris.
destinations = []
unresolved_cities = []

for idee in idees_vacances:
    city_name = idee["city"]
    country_code = idee["country"]

    matching_cities = cities[
        cities["_normalized_name"] == normalize_name(city_name)
    ].copy()

    if country_code:
        matching_cities = matching_cities[
            matching_cities["country_code"] == country_code
        ]

    if matching_cities.empty:
        unresolved_cities.append(
            f"{city_name}"
            + (
                f" ({country_code})"
                if country_code else ""
            )
        )
        continue

    # En cas de ville homonyme sans code pays,
    # la ville la plus peuplée est retenue.
    selected_city = matching_cities.sort_values(
        "population",
        ascending=False,
    ).iloc[0]

    resolved_country = str(
        selected_city["country_code"]
    )

    destinations.append({
        "city": str(selected_city["name"]),
        "country": resolved_country,
        "latitude": float(selected_city["latitude"]),
        "longitude": float(selected_city["longitude"]),
    })


if unresolved_cities:
    st.warning(
        "Certaines villes n'ont pas pu être retrouvées : "
        + ", ".join(unresolved_cities)
        + ". Tu peux les réenregistrer depuis Inspirations de voyage."
    )


if not destinations:
    st.error(
        "Aucune ville enregistrée n'a pu être retrouvée."
    )
    st.stop()


# --------------------------------------------------
# Regroupement des destinations par pays
# --------------------------------------------------

destinations_by_country = {}

for destination in destinations:
    code = destination["country"]

    if code not in destinations_by_country:
        destinations_by_country[code] = []

    destinations_by_country[code].append(destination)


# Pays triés par ordre alphabétique
country_codes = sorted(
    destinations_by_country,
    key=lambda code: normalize_name(
        countries.get(code, code)
    ),
)


# --------------------------------------------------
# Affichage : pays > villes > lieux
# --------------------------------------------------

total_cities = 0
total_places = 0

for country_code in country_codes:
    country_name = countries.get(
        country_code,
        country_code,
    )

    country_destinations = sorted(
        destinations_by_country[country_code],
        key=lambda destination: normalize_name(
            destination["city"]
        ),
    )

    city_count = len(country_destinations)

    # Un expander par pays
    with st.expander(
        f"🌍 {country_name} — {city_count} "
        f"ville(s)",
        expanded=False,
    ):

        for destination in country_destinations:
            city_name = destination["city"]
            latitude = destination["latitude"]
            longitude = destination["longitude"]

            total_cities += 1

            # Un expander par ville à l'intérieur du pays
            with st.expander(
                f"📍 {city_name}",
                expanded=False,
            ):

                nearby = places.copy()

                nearby["distance_km"] = distances_from_point(
                    nearby,
                    latitude,
                    longitude,
                )

                # Rayon strict : aucun élargissement automatique.
                nearby = nearby[
                    nearby["distance_km"] <= RADIUS_KM
                ].copy()

                nearby = rank_places(nearby)
                top_places = nearby.head(MAX_PLACES)

                st.caption(
                    f"Les {MAX_PLACES} meilleurs lieux au maximum "
                    f"dans un rayon de {RADIUS_KM} km."
                )

                st.write(
                    f"**{len(top_places)} lieu(x) trouvé(s)**"
                )

                if top_places.empty:
                    st.info(
                        "Aucun lieu enregistré dans ce rayon."
                    )
                    continue

                total_places += len(top_places)

                for position, (_, place) in enumerate(
                    top_places.iterrows(),
                    start=1,
                ):

                    with st.container(border=True):

                        col_info, col_image = st.columns(
                            [2, 1],
                            gap="medium",
                        )

                        with col_info:
                            st.markdown(
                                f"### {position}. {place['name']}"
                            )

                            st.caption(
                                f"Distance : "
                                f"{place['distance_km']:.1f} km"
                            )

                            if (
                                "score" in top_places.columns
                                and pd.notna(place.get("score"))
                            ):
                                st.write(
                                    f"**Score :** {place['score']}"
                                )

                            elif (
                                "fame_rank" in top_places.columns
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
                                st.write(
                                    f"**Type :** {kind_label}"
                                )

                            description = place.get("description")

                            if (
                                pd.notna(description)
                                and str(description).strip()
                            ):
                                st.write(
                                    str(description).strip()
                                )

                            links = []

                            wikipedia_url = place.get("wikipedia")
                            map_url = place.get("map_url")

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
                                st.caption(
                                    "Aucune photo disponible."
                                )

