import streamlit as st
import pandas as pd
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
client_credentials_manager = SpotifyClientCredentials(
    client_id='1e10e9e3ab3844a8acf959b9fdaa86df', client_secret='babcb04988054942a225ebf446f0ba6f')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

data = pd.read_csv("dataframe_with_more_keys.csv")


def fetch_song_features(song_name, artist_name):
    # Search for the song using name and artist
    results = sp.search(
        q=f'track:{song_name} artist:{artist_name}', type='track', limit=1)

    if results['tracks']['items']:
        track_info = results['tracks']['items'][0]
        track_id = track_info['id']

        # Fetch audio features for the track
        audio_features = sp.audio_features(track_id)

        if audio_features:
            return audio_features[0]

    return None


def nearest_neighbours(df, song_name, artist_name):
    numeric_data = df.drop(
        ['Name', 'Artist', 'Album', 'Popularity', 'Lyrics', 'genre'], axis=1)

    # User-entered song features (obtained using fetch_song_features)
    user_song_features = fetch_song_features(song_name, artist_name)

    # Create a DataFrame for the user's song with only numeric features
    user_song_df = pd.DataFrame(
        [user_song_features], columns=numeric_data.columns)

    # Calculate the nearest neighbors
    k = 5  # Number of nearest neighbors to find
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm='ball_tree').fit(numeric_data)
    distances, indices = nbrs.kneighbors(user_song_df)

    # Get the 5 nearest songs based on indices
    nearest_songs = df.iloc[indices[0]]
    return nearest_neighbours


def find_songs_with_genres(dataframe, genres_to_find):
    matching_songs = []

    for index, row in dataframe.iterrows():
        song_genres = row['genre']
        if song_genres is not None and any(genre in song_genres for genre in genres_to_find):
            matching_songs.append(row)

    if matching_songs:
        matching_songs_df = pd.DataFrame(matching_songs)
        return matching_songs_df
    else:
        return pd.DataFrame()


def get_artist_genre(artist_name):
    # Use the Spotify API to get the genre of the artist
    artist = sp.search(q=artist_name, type='artist')
    if 'artists' in artist and 'items' in artist['artists'] and len(artist['artists']['items']) > 0:
        genres = artist['artists']['items'][0]['genres']
        if genres:
            return genres
    return None


def recommend_song_genre_neighbour(song_name, artist_name):
    genre = get_artist_genre(artist_name)

    genre_df = find_songs_with_genres(data, genre)
    print()

    top5 = nearest_neighbours(genre_df, song_name, artist_name)

    return top5


# Page configuration
st.set_page_config(
    page_title="Song Recommender",
    layout="centered"
)

# Streamlit App Title
st.title("Song Recommender")

# User Input Section
st.header("Enter the Name of the Artist and the Song")

# Input fields for artist and song
artist_name = st.text_input("Artist Name")
song_name = st.text_input("Song Name")

# Button to trigger song recommendation
if st.button("Recommend Songs"):
    st.write(f"Recommendations for {song_name} by {artist_name}:")

    # Call the recommend_song_genre_neighbour function
    top5 = recommend_song_genre_neighbour(song_name, artist_name)

    if not top5.empty:
        st.write(top5)
    else:
        st.write("No recommendations found based on genre match.")

# Footer
st.text("© 2023 YourWebsite.com")

# If you need to hide the Streamlit menu, uncomment the line below
# st.markdown('<style>div.Widget.row-widget.stRadio > div{display:none;}</style>', unsafe_allow_html=True)
