import json
import unicodedata
import re
from pathlib import Path

import requests
import streamlit as st


# --------------------------------------------------
# Fichiers
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

VISITED_FILE = BASE_DIR / "visited_countries.json"
MEMORIES_FILE = BASE_DIR / "travel_memories.json"


# --------------------------------------------------
# Source cartographique
# --------------------------------------------------

GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/"
    "natural-earth-vector/master/geojson/"
    "ne_110m_admin_0_countries.geojson"
)


# --------------------------------------------------
# Pays visités par défaut
# --------------------------------------------------

DEFAULT_VISITED = {"FRA", "BEL", "ESP", "ITA", "DEU"}


# --------------------------------------------------
# Pays supplémentaires
# --------------------------------------------------

# Ces codes sont utilisés par l'application.
# SCO est un code interne pour l'Écosse, pas un code ISO officiel.
EXTRA_COUNTRIES = {
    "MLT": "Malte",
    "VAT": "Vatican",
    "SCO": "Écosse",
}


# --------------------------------------------------
# Fonctions JSON
# --------------------------------------------------

def read_json(path, default):
    """Lit un fichier JSON ou renvoie une valeur par défaut."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    return default


def write_json(path, data):
    """Enregistre des données au format JSON."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------
# Pays visités
# --------------------------------------------------

def load_visited():
    """Renvoie l'ensemble des codes des pays visités."""
    return set(read_json(VISITED_FILE, sorted(DEFAULT_VISITED)))


def save_visited(visited):
    """Sauvegarde les pays visités."""
    write_json(VISITED_FILE, sorted(visited))


# --------------------------------------------------
# Souvenirs (conservés pour compatibilité)
# --------------------------------------------------

def load_memories():
    """Charge les souvenirs associés à chaque pays."""
    return read_json(MEMORIES_FILE, {})


def save_memories(memories):
    """Sauvegarde les souvenirs associés à chaque pays."""
    write_json(MEMORIES_FILE, memories)


# --------------------------------------------------
# Normalisation des noms
# --------------------------------------------------

def normalize_name(value):
    """Normalise un nom pour faciliter les comparaisons."""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    return re.sub(r"[^a-z0-9]", "", value)


# --------------------------------------------------
# Chargement des pays et des frontières
# --------------------------------------------------

@st.cache_data
def load_countries():
    """
    Télécharge les frontières des pays et prépare le catalogue.

    Malte et le Vatican sont ajoutés au catalogue si nécessaire.
    L'Écosse est ajoutée comme entrée indépendante pour la liste.
    """

    response = requests.get(GEOJSON_URL, timeout=30)
    response.raise_for_status()

    geojson = response.json()
    countries = {}

    # Noms pouvant être rencontrés dans les données cartographiques.
    aliases = {
        "malta": ("MLT", "Malte"),
        "vatican": ("VAT", "Vatican"),
        "vaticancity": ("VAT", "Vatican"),
        "holysee": ("VAT", "Vatican"),
        "scotland": ("SCO", "Écosse"),
    }

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})

        name = (
            props.get("ADMIN")
            or props.get("NAME")
            or props.get("NAME_EN")
            or ""
        )

        normalized_name = normalize_name(name)

        # Détection spéciale par nom.
        # Cela permet de corriger certains codes absents du GeoJSON.
        if normalized_name in aliases:
            code, display_name = aliases[normalized_name]
        else:
            code = props.get("ISO_A3_EH")

            if not code or code == "-99":
                code = props.get("ISO_A3")

            if not code or code == "-99":
                code = props.get("ADM0_A3")

            if not code or code == "-99":
                continue

            display_name = name or code

        # Ces propriétés servent à identifier les pays sur la carte.
        props["_app_iso"] = code
        props["_app_name"] = display_name

        countries[code] = display_name

    # Garantit que les trois entrées existent dans la liste,
    # même si le GeoJSON ne fournit pas leurs frontières.
    countries.update(EXTRA_COUNTRIES)

    return geojson, countries
