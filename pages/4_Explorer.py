import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PLACES_FILE = BASE_DIR / "places.csv"
CITIES_FILE = BASE_DIR / "cities15000.txt"
COUNTRIES_FILE = BASE_DIR / "countryInfo.txt"
IDEAS_FILE = BASE_DIR / "idees_vacances.json"

RADIUS_KM = 5
EARTH_RADIUS_KM = 6371

st.set_page_config(
    page_title="Explorer des destinations",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Explorer des destinations")
st.write(
    "Trouve des villes proches de paysages naturels spectaculaires "
    "et de sites historiques."
)

NATURE_THEMES = [
    "Montagnes",
    "Merveilles naturelles",
    "Cascades",
    "Plages",
]

HISTORY_THEMES = [
    "Châteaux",
    "Ruines antiques",
]

ALL_THEMES = HISTORY_THEMES + NATURE_THEMES

CONTINENTS = {
    "Afrique": "AF",
    "Asie": "AS",
    "Europe": "EU",
    "Amérique du Nord": "NA",
    "Amérique du Sud": "SA",
    "Océanie": "OC",
}


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def normalize(value):
    """Normalise un texte pour faciliter les comparaisons."""
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return value.strip()


def valid_url(value):
    """N'accepte que les URL HTTP(S)."""
    if not isinstance(value, str):
        return None

    value = value.strip()

    if value.startswith(("https://", "http://")):
        return value

    return None


def spherical_coordinates(latitudes, longitudes):
    """Convertit latitude/longitude en coordonnées cartésiennes."""
    latitudes = np.radians(np.asarray(latitudes, dtype=float))
    longitudes = np.radians(np.asarray(longitudes, dtype=float))

    x = np.cos(latitudes) * np.cos(longitudes)
    y = np.cos(latitudes) * np.sin(longitudes)
    z = np.sin(latitudes)

    return np.column_stack((x, y, z))


def distance_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique entre deux points."""
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# ============================================================
# CLASSIFICATION DES LIEUX
# ============================================================

def classify_place(row):
    """
    Renvoie une liste de thèmes pour chaque lieu.

    Un même lieu peut correspondre à plusieurs thèmes.
    Par exemple, un parc national montagneux peut correspondre
    à « Montagnes » et « Merveilles naturelles ».
    """
    text = row.get("_search_text", "")

    if not text:
        text = normalize(
            " ".join(
                str(row.get(col, ""))
                for col in [
                    "name",
                    "kind",
                    "category",
                    "description",
                ]
            )
        )

    groups = [
        (
            "Châteaux",
            [
                r"\bcastle\b",
                r"\bchateau\b",
                r"\bchateaux\b",
                r"\bcastello\b",
                r"\bburg\b",
                r"\bschloss\b",
                r"\bcastillo\b",
                r"\bfortress\b",
            ],
        ),
        (
            "Ruines antiques",
            [
                r"\bancient ruin",
                r"\broman ruin",
                r"\bgreek ruin",
                r"\barchaeological site\b",
                r"\barcheological site\b",
                r"\bancient temple\b",
                r"\broman theatre\b",
                r"\broman theater\b",
                r"\broman city\b",
                r"\bancient city\b",
                r"\bantique ruin",
                r"\bruins of\b",
                r"\barchaeological ruins\b",
            ],
        ),
        (
            "Cascades",
            [
                r"\bwaterfall",
                r"\bwaterfalls\b",
                r"\bcascade\b",
                r"\bcascades\b",
                r"\bfalls\b",
                r"\bchute d eau\b",
                r"\bchutes d eau\b",
            ],
        ),
        (
            "Plages",
            [
                r"\bbeach\b",
                r"\bbeaches\b",
                r"\bplage\b",
                r"\bplages\b",
                r"\bsandy beach\b",
                r"\bwhite sand\b",
            ],
        ),
        (
            "Montagnes",
            [
                r"\bmountain\b",
                r"\bmountains\b",
                r"\bmount\b",
                r"\bmontagne\b",
                r"\bmontagnes\b",
                r"\bmountain range\b",
                r"\bmountain peak\b",
                r"\bmountain summit\b",
                r"\bmassif\b",
                r"\bpeak\b",
                r"\bsummit\b",
            ],
        ),
        (
            "Merveilles naturelles",
            [
                r"\bnatural wonder\b",
                r"\bnatural wonders\b",
                r"\bnational park\b",
                r"\bnature reserve\b",
                r"\bnatural monument\b",
                r"\bgeological formation\b",
                r"\bcanyon\b",
                r"\bcavern\b",
                r"\bcave\b",
                r"\bgorge\b",
                r"\bcliff\b",
                r"\bvolcano\b",
                r"\bvolcanic\b",
                r"\bglacier\b",
                r"\bgeyser\b",
                r"\bkarst\b",
                r"\brock formation\b",
                r"\bvalley\b",
                r"\bdesert\b",
                r"\bforest\b",
                r"\brainforest\b",
            ],
        ),
    ]

    matched_themes = []

    for theme, patterns in groups:
        if any(re.search(pattern, text) for pattern in patterns):
            matched_themes.append(theme)

    return matched_themes


# ============================================================
# CHARGEMENT DES PAYS
# ============================================================

@st.cache_data
def load_countries():
    if not COUNTRIES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {COUNTRIES_FILE}"
        )

    columns = [
        "iso",
        "iso3",
        "iso_numeric",
        "fips",
        "country",
        "capital",
        "area",
        "population",
        "continent",
        "tld",
        "currency_code",
        "currency_name",
        "phone",
        "postal_code_format",
        "postal_code_regex",
        "languages",
        "geonameid",
        "neighbours",
        "equivalent_fips",
    ]

    df = pd.read_csv(
        COUNTRIES_FILE,
        sep="\t",
        comment="#",
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
    )

    for col in ["country", "iso", "continent"]:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df["country"] != "")
        & (df["iso"] != "")
        & df["continent"].isin(CONTINENTS.values())
    ].copy()

    return df


# ============================================================
# CHARGEMENT DES VILLES
# ============================================================

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

    df = pd.read_csv(
        CITIES_FILE,
        sep="\t",
        header=None,
        names=columns,
        usecols=[
            "name",
            "latitude",
            "longitude",
            "country_code",
            "population",
        ],
        dtype={
            "name": str,
            "country_code": str,
        },
        low_memory=False,
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )
    df["population"] = pd.to_numeric(
        df["population"], errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=["name", "latitude", "longitude"]
    ).copy()

    df["country_code"] = (
        df["country_code"].astype(str).str.strip()
    )

    # Exclure les arrondissements de Paris, Marseille et Lyon.
    district_pattern = (
        r"\bParis\s+\d+\b"
        r"|\bMarseille\s+\d+\b"
        r"|\bLyon\s+\d+\b"
        r"|\b\d+(?:er|e|eme|ème)\s+Arrondissement\b"
    )

    districts = df["name"].str.contains(
        district_pattern,
        case=False,
        regex=True,
        na=False,
    )

    df = df[~districts].copy()

    return df


# ============================================================
# CHARGEMENT DES LIEUX
# ============================================================

@st.cache_data
def load_places():
    if not PLACES_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PLACES_FILE}"
        )

    columns = [
        "id",
        "name",
        "kind",
        "category",
        "country",
        "country_code",
        "latitude",
        "longitude",
        "field_8",
        "field_9",
        "fame_rank",
        "field_11",
        "field_12",
        "wikipedia",
        "image_url",
        "website",
        "description",
    ]

    df = pd.read_csv(
        PLACES_FILE,
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
        low_memory=False,
    )

    for col in ["latitude", "longitude", "fame_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    df["country_code"] = (
        df["country_code"].astype(str).str.strip()
    )

    text_columns = [
        "name",
        "kind",
        "category",
        "description",
    ]

    df["_search_text"] = df[text_columns].fillna("").apply(
        lambda row: normalize(" ".join(row.astype(str))),
        axis=1,
    )

    # Chaque lieu peut correspondre à plusieurs thèmes.
    df["_themes"] = df.apply(classify_place, axis=1)

    # On conserve les lieux correspondant à au moins un thème.
    df = df[df["_themes"].map(bool)].copy()

    return df


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

try:
    countries_df = load_countries()
    cities_df = load_cities()
    places_df = load_places()

except Exception as error:
    st.error(f"Impossible de charger les données : {error}")
    st.stop()


# ============================================================
# FILTRES
# ============================================================

st.subheader("1. Où veux-tu chercher ?")

continent_name = st.selectbox(
    "Continent",
    options=list(CONTINENTS.keys()),
)

continent_code = CONTINENTS[continent_name]

continent_countries = countries_df[
    countries_df["continent"] == continent_code
].sort_values("country")

country_options = continent_countries["country"].tolist()

selected_countries = st.multiselect(
    "Pays à explorer (maximum 10)",
    options=country_options,
    max_selections=10,
    help="Choisis entre 1 et 10 pays sur le continent sélectionné.",
)

st.subheader("2. Choisis les thèmes")

col_history, col_nature = st.columns(2)

with col_history:
    st.markdown("#### 🏰 Histoire")

    selected_history = st.multiselect(
        "Sites historiques",
        options=HISTORY_THEMES,
        default=["Châteaux"],
        key="selected_history",
    )

with col_nature:
    st.markdown("#### 🏔️ Paysages naturels")

    selected_nature = st.multiselect(
        "Paysages naturels",
        options=NATURE_THEMES,
        default=["Montagnes"],
        key="selected_nature",
    )

selected_themes = selected_history + selected_nature

st.caption(
    f"Une ville doit avoir au moins un lieu correspondant à "
    f"chaque thème sélectionné dans un rayon de {RADIUS_KM} km. "
    "Le classement repose sur le rang des lieux correspondants."
)

search_clicked = st.button(
    "🔎 Rechercher les destinations",
    type="primary",
    use_container_width=True,
)


# ============================================================
# CALCUL DES RÉSULTATS
# ============================================================

def calculate_city_results(
    country_cities,
    country_places,
    themes,
):
    """
    Une ville est retenue si tous les thèmes sélectionnés
    sont présents dans le rayon défini.

    Score d'un lieu :
        nombre de thèmes sélectionnés satisfaits / fame_rank

    Score de la ville :
        somme des scores des lieux correspondants.
    """
    if country_cities.empty or country_places.empty or not themes:
        return []

    selected_themes_set = set(themes)

    # Garder les lieux correspondant à au moins un filtre.
    eligible_places = country_places[
        country_places["_themes"].map(
            lambda place_themes: bool(
                selected_themes_set.intersection(place_themes)
            )
        )
    ].copy()

    if eligible_places.empty:
        return []

    place_xyz = spherical_coordinates(
        eligible_places["latitude"].to_numpy(),
        eligible_places["longitude"].to_numpy(),
    )

    tree = cKDTree(place_xyz)

    # Conversion du rayon en corde sphérique.
    radius_chord = 2 * np.sin(
        (RADIUS_KM / EARTH_RADIUS_KM) / 2
    )

    results = []

    for _, city in country_cities.iterrows():
        city_xyz = spherical_coordinates(
            [city["latitude"]],
            [city["longitude"]],
        )[0]

        indices = tree.query_ball_point(
            city_xyz,
            r=radius_chord,
        )

        if not indices:
            continue

        nearby = eligible_places.iloc[indices].copy()

        nearby["_distance_km"] = distance_km(
            city["latitude"],
            city["longitude"],
            nearby["latitude"].to_numpy(),
            nearby["longitude"].to_numpy(),
        )

        nearby = nearby[
            nearby["_distance_km"] <= RADIUS_KM
        ].copy()

        if nearby.empty:
            continue

        # Thèmes sélectionnés réellement satisfaits par chaque lieu.
        nearby["_matched_themes"] = nearby["_themes"].map(
            lambda place_themes: sorted(
                selected_themes_set.intersection(place_themes)
            )
        )

        # Nombre de filtres satisfaits par chaque lieu.
        nearby["_matched_count"] = nearby[
            "_matched_themes"
        ].map(len)

        # Contribution au score : N / rang.
        # Un rang absent ou inférieur ou égal à zéro vaut zéro.
        valid_rank = (
            nearby["fame_rank"].notna()
            & (nearby["fame_rank"] > 0)
        )

        nearby["_score_contribution"] = 0.0

        nearby.loc[valid_rank, "_score_contribution"] = (
            nearby.loc[valid_rank, "_matched_count"]
            / nearby.loc[valid_rank, "fame_rank"]
        )

        # Vérifier que chaque thème choisi est représenté.
        found_themes = set(
            theme
            for matched in nearby["_matched_themes"]
            for theme in matched
        )

        if not selected_themes_set.issubset(found_themes):
            continue

        # Le score de la ville est la somme des contributions.
        score = float(nearby["_score_contribution"].sum())

        # Affichage : d'abord les lieux les plus contributeurs.
        nearby = nearby.sort_values(
            by=[
                "_score_contribution",
                "fame_rank",
                "_distance_km",
            ],
            ascending=[False, True, True],
            na_position="last",
        )

        population = int(city["population"])
        place_count = len(nearby)

        results.append(
            {
                "city": city["name"],
                "latitude": float(city["latitude"]),
                "longitude": float(city["longitude"]),
                "population": population,
                "place_count": place_count,
                "score": score,
                "themes": sorted(found_themes),
                "places": nearby,
            }
        )

    # Classement par score décroissant.
    results.sort(
        key=lambda item: (
            item["score"],
            item["place_count"],
        ),
        reverse=True,
    )

    return results


def run_search(countries, themes):
    all_results = {}

    for country_name in countries:
        country_row = countries_df[
            countries_df["country"] == country_name
        ]

        if country_row.empty:
            all_results[country_name] = []
            continue

        country_code = country_row.iloc[0]["iso"]

        country_cities = cities_df[
            cities_df["country_code"] == country_code
        ]

        country_places = places_df[
            places_df["country_code"] == country_code
        ]

        all_results[country_name] = calculate_city_results(
            country_cities,
            country_places,
            themes,
        )

    return all_results


if search_clicked:
    if not selected_countries:
        st.warning("Sélectionne au moins un pays.")
        st.session_state.pop("explorer_results", None)

    elif not selected_themes:
        st.warning("Sélectionne au moins un thème.")
        st.session_state.pop("explorer_results", None)

    else:
        with st.spinner("Recherche des destinations en cours…"):
            st.session_state["explorer_results"] = run_search(
                selected_countries,
                selected_themes,
            )
            st.session_state["explorer_themes"] = selected_themes.copy()
            st.session_state["explorer_countries"] = selected_countries.copy()


# ============================================================
# AJOUT AUX IDÉES DE VACANCES
# ============================================================

def add_to_ideas(result, country_name):
    """Ajoute une destination au fichier idees_vacances.json."""
    if IDEAS_FILE.exists():
        try:
            with open(IDEAS_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            data = []
    else:
        data = []

    if isinstance(data, dict):
        ideas = data.setdefault("idees", [])
    elif isinstance(data, list):
        ideas = data
    else:
        ideas = []

    city_name = result["city"]

    exists = any(
        isinstance(item, dict)
        and normalize(item.get("ville", item.get("city", "")))
        == normalize(city_name)
        and normalize(item.get("pays", item.get("country", "")))
        == normalize(country_name)
        for item in ideas
    )

    if exists:
        return False

    ideas.append(
        {
            "ville": city_name,
            "pays": country_name,
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "population": result["population"],
            "themes": result["themes"],
            "lieux": [
                str(name)
                for name in result["places"]["name"].tolist()
            ],
        }
    )

    if isinstance(data, dict):
        data["idees"] = ideas
        output = data
    else:
        output = ideas

    with open(IDEAS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return True


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

all_results = st.session_state.get("explorer_results")

# Invalider les anciens résultats qui ne contiennent pas
# les colonnes nécessaires à l'affichage.
if all_results is not None:
    results_are_valid = True

    for city_results in all_results.values():
        for city_result in city_results:
            places = city_result.get("places")

            if places is None or not {
                "_matched_themes",
                "_distance_km",
                "_score_contribution",
            }.issubset(places.columns):
                results_are_valid = False
                break

        if not results_are_valid:
            break

    if not results_are_valid:
        st.session_state.pop("explorer_results", None)
        st.session_state.pop("explorer_themes", None)
        st.session_state.pop("explorer_countries", None)
        st.rerun()

if all_results is not None:
    total_cities = sum(
        len(results) for results in all_results.values()
    )

    st.divider()
    st.subheader("🗺️ Résultats")

    st.metric(
        "Villes correspondant à tous les thèmes",
        total_cities,
    )

    if total_cities == 0:
        st.warning(
            "Aucune ville ne correspond à tous les thèmes "
            "sélectionnés dans les pays choisis. Essaie de "
            "sélectionner moins de thèmes ou d'autres pays."
        )

    for country_name, results in all_results.items():
        with st.expander(
            f"🌐 {country_name} — {len(results)} villes",
            expanded=(len(results) > 0),
        ):
            if not results:
                st.info(
                    "Aucune ville ne correspond à tous les thèmes "
                    "sélectionnés dans ce pays."
                )
                continue

            st.caption(
                "Classement par somme des contributions des lieux : "
                "nombre de thèmes satisfaits divisé par le rang."
            )

            selected_city_index = st.selectbox(
                "Choisir une ville pour voir ses lieux",
                options=list(range(len(results))),
                format_func=lambda i: (
                    f"{i + 1}. {results[i]['city']} — "
                    f"{results[i]['place_count']} lieux — "
                    f"score {results[i]['score']:.5f}"
                ),
                key=f"city_select_{normalize(country_name)}",
            )

            result = results[selected_city_index]

            st.markdown("---")
            st.subheader(
                f"📍 {selected_city_index + 1}. {result['city']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Population",
                    f"{result['population']:,}".replace(",", " "),
                )

            with col2:
                st.metric(
                    "Lieux à proximité",
                    result["place_count"],
                )

            with col3:
                st.metric(
                    "Score",
                    f"{result['score']:.5f}",
                )

            st.markdown("**Thèmes trouvés**")
            st.write(", ".join(result["themes"]))

            map_url = (
                "https://www.openstreetmap.org/"
                f"?mlat={result['latitude']}"
                f"&mlon={result['longitude']}"
                f"#map=11/{result['latitude']}/{result['longitude']}"
            )

            st.markdown(
                f"[🗺️ Voir {result['city']} sur OpenStreetMap]"
                f"({map_url})"
            )

            st.markdown(
                f"**Lieux situés dans un rayon de {RADIUS_KM} km**"
            )

            for _, place in result["places"].iterrows():
                place_name = str(place["name"])
                place_distance = float(place["_distance_km"])
                matched_themes = place["_matched_themes"]
                rank = place["fame_rank"]
                contribution = float(place["_score_contribution"])

                with st.container():
                    st.markdown(f"**{place_name}**")

                    st.caption(
                        f"{' · '.join(matched_themes)} · "
                        f"{place_distance:.1f} km de la ville"
                    )

                    if pd.notna(rank) and rank > 0:
                        st.caption(
                            f"Rang : {rank:g} · "
                            f"Contribution au score : {contribution:.6f}"
                        )
                    else:
                        st.caption(
                            "Rang indisponible : contribution au score nulle."
                        )

                    description = str(
                        place.get("description", "")
                    ).strip()

                    if description and description.lower() != "nan":
                        st.write(description)

                    image_url = valid_url(
                        place.get("image_url", "")
                    )

                    if image_url:
                        st.markdown(
                            f"[Voir l'image du lieu]({image_url})"
                        )

                    wikipedia_url = valid_url(
                        place.get("wikipedia", "")
                    )

                    website_url = valid_url(
                        place.get("website", "")
                    )

                    links = []

                    if wikipedia_url:
                        links.append(f"[Wikipedia]({wikipedia_url})")

                    if website_url:
                        links.append(f"[Site officiel]({website_url})")

                    if links:
                        st.markdown(" · ".join(links))

                    st.divider()

            if st.button(
                "➕ Ajouter aux idées de vacances",
                key=(
                    f"add_idea_{normalize(country_name)}_"
                    f"{selected_city_index}"
                ),
                use_container_width=True,
            ):
                try:
                    added = add_to_ideas(result, country_name)

                    if added:
                        st.success(
                            f"{result['city']} a été ajoutée "
                            "à tes idées de vacances."
                        )
                    else:
                        st.info(
                            "Cette destination est déjà présente "
                            "dans tes idées de vacances."
                        )

                except OSError as error:
                    st.error(
                        f"Impossible d'enregistrer la destination : {error}"
                    )