from ..MusicInfluenceNetwork.ArtistsClients import MusicbrainzArtistClient

def main() :
    genre = "rock"
    num_artists = 1000
    limit = 100

    client = MusicbrainzArtistClient()
    #artists = client.artists_genre(genre, num_artists, limit, is_sleep=True)
    artists_stream = client.stream_artists_genre(genre, num_artists, limit, is_sleep=True)
    print(artists_stream)

if __name__ == '__main__':
    main()