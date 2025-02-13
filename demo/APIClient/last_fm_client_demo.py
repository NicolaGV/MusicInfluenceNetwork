import os
import json
import yaml
import dataclasses

from dotenv import load_dotenv

from MusicInfluenceNetwork.Clients import LastFMClient

""" Alternative JSON encoder to serialize dataclasses and avoid converting them to dictionaries
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
"""

def main():
    
    load_dotenv()
    
    api_key = os.getenv("LAST_FM_API_KEY")    
    
    client = LastFMClient(api_key)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Example artists list
    with open(f'{script_dir}/top_10_music_artists.txt', 'r') as f:
        artists = f.readlines()
        
    artists = [artist.strip() for artist in artists]
    
    similar_dict = {}
    
    for artist in artists:
        similar_dict[artist] = client.get_similar_artists(artist)
    
    similar_dict_converted = {k: [dataclasses.asdict(v) for v in vs] for k, vs in similar_dict.items()}
    
    # Example json dump
    with open(f'{script_dir}/similar_artists.json', 'w') as f:
        json.dump(similar_dict_converted, f, indent=4)
        
    # Example yaml dump
    with open(f'{script_dir}/similar_artists.yaml', 'w') as f:
        yaml.dump(similar_dict_converted, f)


if __name__ == '__main__':
    main()