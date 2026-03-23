import pickle
import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
OMDB_API_KEY = "c452a97d"

# --- SESSION STATE ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_search' not in st.session_state:
    st.session_state.current_search = ""

# --- DATA LOADING ---
@st.cache_resource
def load_local_data():
    try:
        with open('movie_dict.pkl', 'rb') as f:
            data = pickle.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        st.error("Missing movie_dict.pkl! Please ensure it is in the project folder.")
        return None

# --- OMDB API FETCHING ---
def fetch_movie_info(movie_title):
    """Fetches posters and metadata for visual display."""
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url).json()
        if response.get('Response') == 'True':
            plot = response.get('Plot', '')
            genres = response.get('Genre', '')
            actors = response.get('Actors', '')
            # Create rich tags for hero/theme matching
            tags = f"{plot} {genres} {actors}".lower()
            return {
                'title': response.get('Title'),
                'tags': tags,
                'poster': response.get('Poster'),
                'year': response.get('Year')
            }
        return None
    except:
        return None

# --- RECOMMENDATION ENGINE ---
def get_recommendations(movie_title, df):
    # Check if movie exists locally or needs API sync
    if movie_title.lower() not in df['title'].str.lower().values:
        new_data = fetch_movie_info(movie_title)
        if new_data:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            st.toast(f"Synchronized '{new_data['title']}' with system!")
        else:
            return [], df

    # Vectorize for Theme & Hero matching
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    try:
        idx = df[df['title'].str.lower() == movie_title.lower()].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        recs = []
        for i in distances[1:7]: # Get top 6
            title = df.iloc[i[0]].title
            info = fetch_movie_info(title)
            if info:
                recs.append(info)
            else:
                # Fallback if poster is missing
                recs.append({'title': title, 'poster': 'https://via.placeholder.com/300x450?text=No+Poster', 'year': 'N/A'})
        return recs, df
    except:
        return [], df

# --- UI SETUP ---
st.set_page_config(page_title="Netflix Recommender", layout="wide")

# Top Menu
head_left, head_right = st.columns([0.8, 0.2])
with head_left:
    st.title("Netflix Movies Recommendation 🍿")
with head_right:
    with st.popover("⋮ History"):
        if st.session_state.search_history:
            for h in reversed(st.session_state.search_history):
                if st.button(f"🔍 {h}", key=f"hist_{h}"):
                    st.session_state.current_search = h
                    st.rerun()
        else:
            st.write("No history yet.")

df = load_local_data()

if df is not None:
    # Use value from history if clicked
    search_query = st.text_input("Enter movie name:", value=st.session_state.current_search)

    if st.button("Find Recommended Movies"):
        if search_query:
            if search_query not in st.session_state.search_history:
                st.session_state.search_history.append(search_query)
            
            st.session_state.current_search = "" # Reset for next use
            
            with st.spinner('Loading posters and analyzing themes...'):
                recommendations, df = get_recommendations(search_query, df)

            if not recommendations:
                st.error("Movie not found locally or online.")
            else:
                st.subheader(f"Because you liked '{search_query}':")
                
                # Create a grid for posters
                grid = st.columns(3)
                for idx, movie in enumerate(recommendations):
                    with grid[idx % 3]:
                        # Using use_column_width=True for compatibility
                        st.image(movie['poster'], use_column_width=True)
                        st.write(f"**{movie['title']}**")
                        st.caption(f"Year: {movie['year']}")
                        
                        # Google "I'm Feeling Lucky" redirect to Netflix
                        search_term = f"site:netflix.com {movie['title']}".replace(" ", "+")
                        netflix_link = f"https://www.google.com/search?q={search_term}&btnI"
                        st.link_button("🚀 Watch on Netflix", netflix_link)
        else:
            st.warning("Please enter a movie title.")

st.divider()
st.caption("Data synced with OMDb API for latest releases.")
