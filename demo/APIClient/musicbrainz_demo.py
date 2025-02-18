import os

from MusicInfluenceNetwork.ArtistsClients import MusicbrainzArtistClient
from dotenv import load_dotenv

def main() :
    genre = "rock"
    num_artists = 1000
    limit = 100
    
    load_dotenv()
    user_agent = os.getenv("MUSICBRAINZ_USERAGENT")

    client = MusicbrainzArtistClient(user_agent)
    #artists = client.artists_genre(genre, num_artists, limit, is_sleep=True)
    artists_stream = client.stream_artists_genre(genre, num_artists, limit, is_sleep=True)
    print(artists_stream)

if __name__ == '__main__':
    main()