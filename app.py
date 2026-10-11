import pandas as pd
import streamlit as st

from data import load_countries, load_visited, save_visited


st.set_page_config(
    page_title="My Travel Map",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 My Travel Map")

st.markdown(
    """
    **Explore le monde à ta façon.**

    Garde une trace des pays visités, découvre de nouvelles destinations
    et prépare tes prochaines aventures.
    """
)

# --------------------------------------------------
# Chargement des pays
# --------------------------------------------------

try:
    _, countries = load_countries()
except Exception:
    st.error(
        "Impossible de charger la liste des pays. "
        "Vérifie ta connexion Internet ou tes fichiers de données."
    )
    st.stop()

if "visited" not in st.session_state:
    st.session_state.visited = load_visited()

visited = st.session_state.visited

# --------------------------------------------------
# Pays visités
# --------------------------------------------------

st.divider()

visited_names = sorted(
    countries[code]
    for code in visited
    if code in countries
)

st.subheader(f"🌍 Mes pays visités ({len(visited_names)})")

if visited_names:
    st.write(" · ".join(visited_names))
else:
    st.info(
        "Aucun pays visité pour le moment. "
        "Commence par sélectionner des pays sur ta carte !"
    )

# --------------------------------------------------
# Ajouter ou retirer un pays manuellement
# --------------------------------------------------

with st.expander("✏️ Ajouter ou retirer un pays"):
    st.caption(
        "Utile pour les petits pays difficiles à sélectionner sur la carte."
    )

    search = st.text_input(
        "Rechercher un pays",
        placeholder="Ex. : Malte, Vatican, Monaco...",
        key="country_search",
    )

    filtered_countries = sorted(
        [
            (code, name)
            for code, name in countries.items()
            if search.strip().casefold() in name.casefold()
        ],
        key=lambda item: item[1].casefold(),
    )

    if filtered_countries:
        selected_code = st.selectbox(
            "Pays",
            options=[code for code, _ in filtered_countries],
            format_func=lambda code: countries[code],
            key="country_selector",
        )

        is_visited = selected_code in visited

        new_status = st.checkbox(
            "Pays visité",
            value=is_visited,
            key=f"visited_checkbox_{selected_code}",
        )

        if new_status != is_visited:
            if new_status:
                visited.add(selected_code)
            else:
                visited.discard(selected_code)

            save_visited(visited)
            st.rerun()
    else:
        st.warning("Aucun pays ne correspond à ta recherche.")

# --------------------------------------------------
# Navigation
# --------------------------------------------------

st.divider()
st.subheader("Que souhaites-tu faire ?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🗺️ Carte à gratter")
    st.write(
        "Colorie les pays que tu as visités et visualise "
        "tes voyages sur la carte du monde."
    )

    if st.button("Ouvrir la carte", use_container_width=True):
        st.switch_page("pages/1_Carte.py")

with col2:
    st.markdown("### 💡 Inspirations")
    st.write(
        "Explore les villes et découvre des lieux intéressants "
        "dans les pays qui t'attirent."
    )

    if st.button("Trouver des idées", use_container_width=True):
        st.switch_page("pages/2_Inspirations.py")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### ✈️ Prochaines vacances")
    st.write(
        "Retrouve les villes que tu as enregistrées "
        "pour tes prochains voyages."
    )

    if st.button("Préparer mes vacances", use_container_width=True):
        st.switch_page("pages/3_Prochaines_vacances.py")

# --------------------------------------------------
# Top 100 des lieux les plus célèbres
# --------------------------------------------------

st.divider()
st.header("🌍 Top 100 des lieux du monde")

st.write(
    "Les lieux les mieux classés dans ton catalogue, "
    "toutes catégories confondues."
)

@st.cache_data
def load_top_places():
    df = pd.read_csv("places.csv")

    df["fame_rank"] = pd.to_numeric(
        df["fame_rank"],
        errors="coerce",
    )

    places = df.dropna(subset=["fame_rank"]).copy()
    places = places.sort_values("fame_rank")
    places = places.drop_duplicates(subset=["qid"])

    return places.head(100)


try:
    places = load_top_places()

    search_place = st.text_input(
        "🔎 Rechercher un lieu",
        placeholder="Ex. : Eiffel Tower, Taj Mahal...",
        key="place_search",
    )

    if search_place.strip():
        places = places[
            places["name"].astype(str).str.contains(
                search_place.strip(),
                case=False,
                na=False,
                regex=False,
            )
            | places["country"].astype(str).str.contains(
                search_place.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if places.empty:
        st.info("Aucun lieu trouvé.")
    else:
        for _, place in places.iterrows():
            rank = int(place["fame_rank"])
            name = place["name"]
            country = (
                place["country"]
                if pd.notna(place["country"])
                else "Pays inconnu"
            )
            kind = (
                place["kind_label"]
                if pd.notna(place["kind_label"])
                else place["kind"]
            )

            with st.container():
                col1, col2 = st.columns([5, 1])

                with col1:
                    st.markdown(
                        f"**{rank}. [{name}]({place['wikipedia']})**"
                        if pd.notna(place["wikipedia"])
                        and str(place["wikipedia"]).strip()
                        else f"**{rank}. {name}**"
                    )
                    st.caption(f"{country} · {kind}")

                with col2:
                    st.caption(f"Rang {rank}")

except FileNotFoundError:
    st.error("Le fichier places.csv est introuvable.")
except Exception as e:
    st.error(f"Impossible de charger les lieux : {e}")
