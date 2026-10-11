import folium
import streamlit as st
from streamlit_folium import st_folium

from data import load_countries, load_visited, save_visited


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Ma carte du monde",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Ma carte du monde")
st.caption(
    "Clique sur un pays pour le marquer comme visité. "
    "Clique à nouveau dessus pour annuler."
)


# --------------------------------------------------
# Chargement des données
# --------------------------------------------------

geojson, countries = load_countries()

if "visited" not in st.session_state:
    st.session_state.visited = load_visited()

if "last_processed_country" not in st.session_state:
    st.session_state.last_processed_country = None

visited = st.session_state.visited


# --------------------------------------------------
# Création de la carte
# --------------------------------------------------

def style_country(feature):
    properties = feature.get("properties", {})
    code = properties.get("_app_iso")

    return {
        "fillColor": "#20B486" if code in visited else "#E5E7EB",
        "color": "#FFFFFF",
        "weight": 1,
        "fillOpacity": 0.85,
    }


world_map = folium.Map(
    location=[20, 0],
    zoom_start=2,
    min_zoom=2,
    tiles="CartoDB positron",
    world_copy_jump=True,
)

folium.GeoJson(
    geojson,
    name="Pays",
    style_function=style_country,
    highlight_function=lambda feature: {
        "weight": 2,
        "color": "#374151",
        "fillOpacity": 1,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["_app_name"],
        aliases=["Pays :"],
        sticky=True,
    ),
).add_to(world_map)


# --------------------------------------------------
# Affichage
# --------------------------------------------------

result = st_folium(
    world_map,
    key="world_map",
    height=620,
    use_container_width=True,
    returned_objects=["last_active_drawing"],
)


# --------------------------------------------------
# Gestion des clics
# --------------------------------------------------

drawing = result.get("last_active_drawing") if result else None

if drawing:
    properties = drawing.get("properties", {})
    code = properties.get("_app_iso")

    if code in countries:
        # Évite de retraiter le même clic lors d'un rerun.
        if code != st.session_state.last_processed_country:
            st.session_state.last_processed_country = code

            if code in visited:
                visited.remove(code)
            else:
                visited.add(code)

            save_visited(visited)
            st.rerun()

