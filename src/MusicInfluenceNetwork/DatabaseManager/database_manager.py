from typing import Optional
import traceback

import mysql.connector
import mysql.connector.cursor

from MusicInfluenceNetwork.Models import Artist, SimilarArtist

class DatabaseManager:
    def __init__(self, config, pool_size: int = 5):
        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=pool_size,
            pool_reset_session=True,
            **config
        )
        
    def get_connection(self):
        return self.pool.get_connection()
        
    def save_genre(self, genre_name: str) -> int:
        connection = self.get_connection()
        cursor = connection.cursor()
        
        query = "INSERT INTO Genre (genre_name) VALUES (%s)"
        try:
            cursor.execute(query, (genre_name,))
            genre_id = cursor.lastrowid
            connection.commit()
        finally:
            cursor.close()
            connection.close()
        return genre_id
    
    def save_artist(self, artist: Artist, is_weak: bool, genre_id: Optional[int]) -> None:
        connection = self.get_connection()
        cursor = connection.cursor()
        
        start_year = "NULL" if artist.career_start_year is None else artist.career_start_year
        end_year = "NULL" if artist.career_end_year is None else artist.career_end_year
            
        query = f"""INSERT INTO Artist
            (mbid, artist_name, career_start_year, career_end_year, is_weak_node, is_explored)
            VALUES (%s, %s, {start_year}, {end_year}, {is_weak}, False)"""
        try:
            cursor.execute(query, (artist.mbid, artist.name))
            artist_id = cursor.lastrowid
            if genre_id is not None:
                genre_query = f"""INSERT INTO ArtistGenre
                (artist_id, genre_id) VALUES ({artist_id}, {genre_id})
                """
                cursor.execute(genre_query)
                connection.commit()
        except Exception as error:
            print(f"Error when trying to insert {artist}: \n{error}")
        finally:
            cursor.close()
            connection.close()
            
    def save_group(self) -> int:
        connection = self.get_connection()
        cursor = connection.cursor()
        
        query = "INSERT INTO ArtistGroup () VALUES ()"
        try:
            cursor.execute(query)
            connection.commit()
        finally:
            cursor.close()
            connection.close()
        return cursor.lastrowid
    
    def get_all_artist_ids_from_group(self, group_id: int) -> list[int]:
        connection = self.get_connection()
        cursor = connection.cursor()
        
        query = f"""SELECT artist_id FROM Artist WHERE group_id = {group_id}"""
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()
        
        result = [row[0] for row in rows]
        return result
    
    def get_artist_group_by_id(self, artist_id) -> int:
        connection = self.get_connection()
        cursor = connection.cursor()
        
        query = f"""SELECT group_id FROM Artist WHERE artist_id = {artist_id}"""
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()
        
        result = rows[0][0]
        return result
            
    def save_many_artists(self, artists: list[Artist], is_weak: bool, genre_id: Optional[int] = None, fix_names: bool = True) -> None:
        if len(artists) == 0:
            return
        connection = self.get_connection()
        cursor = connection.cursor()
        
        group_id = self.save_group()
        
        start_year_list = []
        end_year_list = []
        mbid_list = []
        [start_year_list.append(start_year) for start_year in
            ["NULL" if artist.career_start_year is None else artist.career_start_year for artist in artists]]
        [end_year_list.append(end_year) for end_year in
            ["NULL" if artist.career_end_year is None else artist.career_end_year for artist in artists]]
        [mbid_list.append(mbid) for mbid in
            ["NULL" if artist.mbid is None else artist.mbid for artist in artists]]
        
        if fix_names:
            for artist in artists:
                artist.name = artist.name.replace("'", "''") # Fix single quote
                artist.name = artist.name.replace("\\", "") # Fix backslash 
        
        query = """INSERT IGNORE INTO Artist  
            (mbid, artist_name, career_start_year, career_end_year, is_weak_node, is_explored, group_id) 
            VALUES """
        for idx, artist in enumerate(artists):
            mbid_insert = f"'{mbid_list[idx]}'" if mbid_list[idx] != "NULL" else "NULL"
            query += f"({mbid_insert}, '{artist.name}', {start_year_list[idx]}, {end_year_list[idx]}, {is_weak}, False, {group_id})"
            if idx < len(artists) - 1:
                query += ", "
        # query += " ON DUPLICATE KEY UPDATE artist_name = artist_name" # Skip duplicate names
        try:
            cursor.execute(query)
            # connection.commit()
            artist_id = cursor.lastrowid
            artists_ids = self.get_all_artist_ids_from_group(group_id)
            if genre_id is not None:
                genre_query = f"""INSERT INTO ArtistGenre
                (artist_id, genre_id) VALUES 
                """
                for idx, artist_id in enumerate(artists_ids):
                    genre_query += f"({artist_id}, {genre_id})"
                    if idx < len(artists_ids) - 1:
                        genre_query += ", "
                cursor.execute(genre_query)
            connection.commit()
        except Exception as error:
            print(f"Error when trying to insert artists:")
            traceback.print_exc()
            raise error
        finally:
            cursor.close()
            connection.close()
            
    def save_similar_artists(self, root_artist_id: Artist, similar_artists: list[SimilarArtist]):
        if len(similar_artists) == 0:
            return
        connection = self.get_connection()
        cursor = connection.cursor()
        
        for artist in similar_artists:
            artist.name = artist.name.replace("\\", "") # Fix backslash 
            artist.name = artist.name.replace("'", r"\'") # Fix single quote
            
        # Insert only artist that are not already in the base
        self.save_many_artists(similar_artists, is_weak=True, fix_names=False)
        
        for artist in similar_artists:
            artist.name = artist.name.replace("\\", "") # Fix backslash 
        
        # Retrieve ids
        artists_names = [artist.name for artist in similar_artists]
        format_strings = ','.join(['%s'] * len(artists_names))
        get_ids_query = f"SELECT artist_id, artist_name FROM Artist WHERE artist_name IN ({format_strings})"
        cursor.execute(get_ids_query, tuple(artists_names))
        rows = cursor.fetchall()
        similar_artists_ids = [row[0] for row in rows]
        similar_artists_names = [row[1] for row in rows]
        
        kept_similar_artists = list(set(filter(lambda artist: artist.name in similar_artists_names, similar_artists)))
        
        similar_insert_query = """INSERT INTO SimilarArtist 
            (artist_id, similar_artist_id, lastFM_similarity, spotify_similarity) 
            VALUES """
        for idx, artist in enumerate(kept_similar_artists):
            similar_insert_query += f"('{root_artist_id}', {similar_artists_ids[idx]}, {artist.last_fm_match}, {artist.spotify_match})"
            if idx < len(kept_similar_artists) - 1:
                similar_insert_query += ", "
        try:
            cursor.execute(similar_insert_query)
            connection.commit()
        finally:
            cursor.close()
            connection.close()
    
    def get_all_artists(self, is_explored: Optional[bool] = None, is_weak: Optional[bool] = None) -> list[Artist]:
        connection = self.get_connection()
        cursor = connection.cursor()
        query = "SELECT * FROM Artist"
        conditions = []
        if is_explored is not None:
            conditions.append(f"is_explored = {is_explored}")
        if is_weak is not None:
            conditions.append(f"is_weak_node = {is_weak}")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            artists = [Artist.from_db_row(row) for row in rows]
        finally:
            cursor.close()
            connection.close()
        return artists
    
    def get_artist_id_from_name(self, artist_name: str) -> int:
        artist_name = artist_name.replace("'", "''") # Fix single quote
        artist_name = artist_name.replace("\\", "") # Fix backslash 
        connection = self.get_connection()
        cursor = connection.cursor()
        query = f"SELECT artist_id FROM Artist WHERE artist_name = '{artist_name}'"
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()
        return rows[0][0]
    
    def update_explored(self, artist_id, is_explored: bool):
        connection = self.get_connection()
        cursor = connection.cursor()
        query = f"""UPDATE Artist SET is_explored = {is_explored} WHERE artist_id = {artist_id}"""
        try:
            cursor.execute(query)
            connection.commit()
        finally:
            cursor.close()
            connection.close()
    
    def empty_database(self) -> None:
        connection = self.get_connection()
        cursor = connection.cursor()
        try:
            response = input("Are you sure you want to empty the database ? (y/n)")
            if response.upper() == "Y":
                cursor.execute("DELETE FROM SimilarArtist")
                cursor.execute("DELETE FROM ArtistGenre")
                cursor.execute("DELETE FROM Genre")
                cursor.execute("DELETE FROM Artist")
                cursor.execute("DELETE FROM ArtistGroup")
                connection.commit()
                print("Database content deleted")
        finally:
            cursor.close()
            connection.close()
    


