import duckdb
# I know these should be run as prepared statements, but I couldn't get it to work, I kept getting told there was a syntax error near '$'

def connect_to_vector_parquet(con: duckdb.DuckDBPyConnection, path: str) -> None:
    con.sql(f"""
            INSTALL vss;
            LOAD vss;
        
            CREATE TABLE vectors AS
            SELECT *
            FROM '{path}';
            """)

def connect_to_concept_csv(con: duckdb.DuckDBPyConnection, path: str) -> None:
    con.sql(f"""
            INSTALL fts;
            LOAD fts;
            
            CREATE TABLE concepts AS
            SELECT *
            FROM '{path}';
            """)
    con.sql("""
            PRAGMA create_fts_index('concepts', 'concept_id', 'concept_name');
            """)
