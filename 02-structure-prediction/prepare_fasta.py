import pandas as pd
import json
import sys
import os

# Define function to convert CSV to FASTA
def csv_to_fasta(csv_path, output_dir, limit):
    # Read the CSV file 
    df = pd.read_csv(csv_path)

    # Take subset for testing container
    subset = df.head(limit) if limit > 0 else df

    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(".fasta"):
                os.remove(os.path.join(output_dir, f))
    os.makedirs(output_dir, exist_ok=True)
    for idx, (_, row) in enumerate(subset.iterrows()):
        fasta_path = os.path.join(output_dir, f"seq_{idx}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">A|protein\n{row['sequence']}\n")
    # Print out the number of sequences written
    print(f"Wrote {len(subset)} sequences to {output_dir}")

if __name__ == "__main__":

    default_config = '{"limit": 0}'
    # Get config from Silva environment
    config_json = sys.argv[1] if len(sys.argv) > 1 else os.getenv('PREDICTION_CONFIG') or default_config

    try:
        config = json.loads(config_json)
        limit = int(config.get('limit', 0))
        input_csv = config.get('input_csv', 'ingested_sequences.csv')
        output_dir = config.get('output', 'fasta_inputs')

        # convert CSV to FASTA
        csv_to_fasta(input_csv, output_dir, limit)
    except json.JSONDecodeError as e:
        print(f"Error: Couldn't parse PREDICTION_CONFIG: {e}")
        sys.exit(1)
