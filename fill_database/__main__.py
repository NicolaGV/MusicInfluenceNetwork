import os

from dotenv import load_dotenv

from MusicInfluenceNetwork.ArtistsClients import MusicbrainzArtistClient
from MusicInfluenceNetwork.DatabaseManager import DatabaseManager
from MusicInfluenceNetwork.Clients import LastFMClient
from MusicInfluenceNetwork.Models import Artist

"""
0. Choisir un genre (metal)

1. Récup des artists des genres récup avant sur MusicBrainz avec weak=False:
    1.1 Pour chaque artist récupéré:
        stocker ses genres dans la table Genre et ArtistGenre

2. Pour chaque artist avec weak=False:
    Récup ses similar artists:
        Si artist connu: tranquille
        Si artist inconnu:
            Si mbid null: ignore artist
            Faire l'étape 1.1 avec cet artiste et weak=True (cad ne pas l'ajouter à la queue pour 2)
"""

db: DatabaseManager


def save_artists_from_genre(mb_user_agent: str, genre: str, nb_artists: int):
    """Step 1 (with only 1 genre)"""
    
    artist_client = MusicbrainzArtistClient(mb_user_agent)
    
    artists = artist_client.get_artists_by_genre(genre, nb_artists, limit=min(nb_artists, 100))
    
    # Remove duplicates based on mbid
    seen_mbids = set()
    unique_artists = []
    for artist in artists:
        if artist.mbid not in seen_mbids:
            unique_artists.append(artist)
            seen_mbids.add(artist.mbid)
    
    genre_id = db.save_genre(genre)
    
    db.save_many_artists(unique_artists, False, genre_id)
        
def save_similar(artist: Artist, last_fm_api_key: str):
    last_fm_client = LastFMClient(last_fm_api_key)
    
    similar_artists = last_fm_client.get_similar_artists(artist.name)
    
    root_id = db.get_artist_id_from_name(artist.name)
    
    db.save_similar_artists(root_id, similar_artists)
    
    db.update_explored(root_id, True)
    
    
def save_similar_of_all_unexplored(last_fm_api_key: str):
    """Step 2 sans refaire 1.1"""
    
    artists = db.get_all_artists(is_explored=False, is_weak=False)
    
    for idx, artist in enumerate(artists):
        save_similar(artist, last_fm_api_key)
        print(f"Saved similar for {idx + 1} / {len(artists)}")
    


def main():
    global db
    load_dotenv()
        
    mb_user_agent = os.getenv("MUSICBRAINZ_USERAGENT")
    
    last_fm_api_key = os.getenv("LAST_FM_API_KEY")
    
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_user = os.getenv("MYSQL_USER")
    mysql_pwd = os.getenv("MYSQL_PWD")
    mysql_db = os.getenv("MYSQL_DB")
    mysql_port = os.getenv("MYSQL_PORT")
    
    config = {
    "host": mysql_host,
    "user": mysql_user,
    "password": mysql_pwd,
    "database": mysql_db,
    "port": mysql_port
    }
    connection_pool_size = 3
    
    db = DatabaseManager(config, pool_size=connection_pool_size)
    
    # db.empty_database()
    
    # Step 1
    # save_artists_from_genre(mb_user_agent, genre="metal", nb_artists=100000)
    
    # Step 2
    # save_similar_of_all_unexplored(last_fm_api_key)
    
    artists = db.get_all_artists()
    
    print(f"Retrieved {len(artists)} artists")
    #[print(artist.name) for artist in artists]

if __name__ == "__main__":
    main()

