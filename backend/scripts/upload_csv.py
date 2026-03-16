import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm

# Add the parent directory to the path so we can import backend.db
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.db.chroma_utils import ChromaCloudDB

def upload_csv(file_path, text_column, type_column, risk_column):
    print("Initializing Database Connection...")
    try:
        db = ChromaCloudDB()
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")
        return

    print(f"\nReading CSV file: {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if text_column not in df.columns:
        print(f"Error: The required text column '{text_column}' was not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    print(f"Found {len(df)} rows to upload.")
    
    # Process each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Uploading to ChromaDB"):
        # 1. Get the text
        text = str(row[text_column])
        
        # Skip empty rows
        if not text or text.lower() == 'nan':
            continue
            
        # 2. Build Metadata
        metadata = {}
        
        if type_column and type_column in df.columns:
            metadata["type"] = str(row[type_column])
        else:
            metadata["type"] = "unknown_text"
            
        if risk_column and risk_column in df.columns:
            metadata["risk_level"] = str(row[risk_column])

        # 3. Create a unique document ID
        # Format: csv_[filename]_[row_number]
        base_filename = os.path.basename(file_path).replace('.csv', '')
        doc_id = f"csv_{base_filename}_row_{index}"

        # 4. Upload to Chroma
        try:
            # We suppress the print statement in add_pattern by temporarily hiding stdout 
            # so it doesn't mess up our tqdm progress bar.
            sys.stdout = open(os.devnull, 'w')
            db.add_pattern(text=text, metadata=metadata, doc_id=doc_id)
            sys.stdout = sys.__stdout__
        except Exception as e:
            sys.stdout = sys.__stdout__
            print(f"\nFailed to upload row {index}: {e}")

    print("\n✅ Bulk upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a CSV of fraud data to ChromaDB.")
    
    # Required Argument
    parser.add_argument("csv_path", help="Path to the CSV file you want to upload.")
    
    # Optional Arguments
    parser.add_argument("--text-col", default="text", help="Name of the column containing the actual message/email. (Default: 'text')")
    parser.add_argument("--type-col", default="type", help="Name of the column identifying if it is an email, sms, etc. (Default: 'type')")
    parser.add_argument("--risk-col", default="risk_level", help="Name of the column identifying risk level. (Default: 'risk_level')")
    
    args = parser.parse_args()
    
    upload_csv(
        file_path=args.csv_path,
        text_column=args.text_col,
        type_column=args.type_col,
        risk_column=args.risk_col
    )
