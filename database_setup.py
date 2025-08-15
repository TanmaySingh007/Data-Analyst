import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import os

def setup_database():
    """
    Set up SQLite database and create table schema
    """
    # Create SQLite database
    db_path = "gaming_data.db"
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Check if dataset exists
    csv_path = "data/online_gaming_behavior.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        print("Please download the dataset first and place it in the data/ directory")
        return None
    
    # Read the CSV file to understand the structure
    print("Reading CSV file to analyze structure...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Create table schema based on the CSV structure
    print("\nCreating database table...")
    
    # Get column types from pandas
    column_definitions = []
    for col, dtype in df.dtypes.items():
        if 'int' in str(dtype):
            sql_type = 'INTEGER'
        elif 'float' in str(dtype):
            sql_type = 'REAL'
        else:
            sql_type = 'TEXT'
        column_definitions.append(f'"{col}" {sql_type}')
    
    # Create table SQL
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS player_behavior (
        {', '.join(column_definitions)}
    )
    """
    
    # Execute table creation
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_behavior"))
        conn.execute(text(create_table_sql))
        conn.commit()
    
    print("Table 'player_behavior' created successfully!")
    
    # Ingest data into the database
    print("Ingesting data into database...")
    df.to_sql('player_behavior', engine, if_exists='replace', index=False)
    
    # Confirm successful ingestion
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM player_behavior"))
        row_count = result.fetchone()[0]
    
    print(f"Successfully ingested {row_count} rows into the database!")
    
    return engine

def load_data_from_sql(engine):
    """
    Load data from SQL database into pandas DataFrame
    """
    print("\nLoading data from SQL database...")
    query = "SELECT * FROM player_behavior"
    df_players = pd.read_sql(query, engine)
    
    print(f"Loaded {len(df_players)} rows into df_players DataFrame")
    return df_players

if __name__ == "__main__":
    # Setup database and ingest data
    engine = setup_database()
    
    if engine:
        # Load data from SQL to pandas
        df_players = load_data_from_sql(engine)
        
        # Save the DataFrame for later use
        df_players.to_pickle("df_players.pkl")
        print("DataFrame saved as 'df_players.pkl'")
    else:
        print("Database setup failed. Please ensure the dataset is available.")
