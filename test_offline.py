import os
import sys
import sqlite3
import pandas as pd
from src.nlp_layer import NLPLayer

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

def run_offline_test():
    db_path = "../data/trends.db"
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}.")
        return

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    try:
        # Load raw data
        query = "SELECT * FROM raw_social_data;"
        df_raw = pd.read_sql_query(query, conn)
        
        if df_raw.empty:
            print("The raw_social_data table is empty. Please run main.py to fetch data first.")
            return
            
        print(f"Successfully loaded {len(df_raw)} raw social media posts.")
        
        # Initialize NLP Layer
        print("Initializing NLP Layer...")
        nlp = NLPLayer()
        
        # Process the dataframe
        df_processed = nlp.process_dataframe(df_raw)
        
        # Extract and print candidate products
        print("\n" + "="*50)
        print("🎯 CANDIDATE PRODUCTS EXTRACTED (OFFLINE TEST) 🎯")
        print("="*50)
        
        all_products = set()
        for idx, row in df_processed.iterrows():
            products = row.get("extracted_products", [])
            if isinstance(products, list):
                for p in products:
                    all_products.add(p)
                    
        if not all_products:
            print("No valid products were extracted after strict filtering.")
        else:
            for i, prod in enumerate(sorted(all_products), 1):
                print(f"{i}. {prod}")
                
        print("="*50)
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_offline_test()
