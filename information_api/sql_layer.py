import os
import psycopg2

#conn_str = os.environ.get('POSTGRESQL_CONN_STRING')
conn_str = "postgres://locindoor:locindoor@locindoordb.crpbxtr1a9mb.us-east-2.rds.amazonaws.com/locindoor"


db_connection = psycopg2.connect(conn_str)


def create_cursor():
    global db_connection
    try:
        return db_connection.cursor()
    except Exception as e:
        db_connection = psycopg2.connect(conn_str)
        return db_connection.cursor()


def create_tables():
    cursor = create_cursor()
    cursor.execute(
        """
        CREATE TABLE pieces(
            piece_id text PRIMARY KEY,
            posifi_id TEXT,
            location_name TEXT,
            description TEXT,
            audio_url TEXT,
            image_url TEXT,
            is_blind_path BOOLEAN
        );
        """
    )
    db_connection.commit()
    cursor.close()


def add_piece(piece_dict):
    cursor = create_cursor()
    cursor.execute(
        """
        INSERT INTO pieces (piece_id, location_name, posifi_id, description, audio_url, image_url, is_blind_path) VALUES(%s, %s, %s, %s, %s, %s, %s);
        """,
        (
            piece_dict['piece_id'],
            piece_dict['location_name'],
            piece_dict['posifi_id'],
            piece_dict['description'],
            piece_dict['audio_url'],
            piece_dict['image_url'],
            'true' if piece_dict['is_blind_path'] else 'false'
        )
    )
    db_connection.commit()
    cursor.close()


def delete_piece(piece_id):
    cursor = create_cursor()
    cursor.execute(
        """
        DELETE FROM pieces WHERE piece_id=%s;
        """,
        (piece_id,)
    )
    db_connection.commit()
    cursor.close()


def edit_piece(piece_id, piece_dict):
    cursor = create_cursor()
    cursor.execute(
        """
        UPDATE pieces SET location_name=%s, posifi_id=%s, description=%s, audio_url=%s, image_url=%s, is_blind_path=%s
        WHERE piece_id = %s;
        """,
        (
            piece_dict['location_name'],
            piece_dict['posifi_id'],
            piece_dict['description'],
            piece_dict['audio_url'],
            piece_dict['image_url'],
            'true' if piece_dict['is_blind_path'] else 'false',
            piece_id
        )
    )
    db_connection.commit()
    cursor.close()


def get_all_pieces():
    cursor = create_cursor()
    cursor.execute("SELECT * FROM pieces;")
    all_pieces = cursor.fetchall()
    pieces = []
    for raw_piece in all_pieces:
        pieces.append(serialize_piece(raw_piece))

    return pieces


def get_pieces_from_location(posifi_id):
    cursor = create_cursor()
    cursor.execute("SELECT * FROM pieces WHERE posifi_id = %s;", (posifi_id,))
    all_pieces = cursor.fetchall()
    pieces = []
    for raw_piece in all_pieces:
        pieces.append(serialize_piece(raw_piece))

    return pieces


def get_piece(piece_id):
    cursor = create_cursor()
    cursor.execute("SELECT * FROM pieces WHERE piece_id = %s;", (piece_id,))
    raw_piece = cursor.fetchone()

    return serialize_piece(raw_piece)


def serialize_piece(raw_piece):
    return {
        "piece_id": raw_piece[0],
        "posifi_id": raw_piece[1],
        "location_name": raw_piece[2],
        "description": raw_piece[3],
        "audio_url": raw_piece[4],
        "image_url": raw_piece[5],
        "is_blind_path": raw_piece[6]
    }
