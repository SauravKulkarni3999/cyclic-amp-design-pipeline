import pandas as pd
import peptides
import json
import sys
import os

# Function for processing sequences
def process_sequences(input_path, output_path):

    if not os.path.exists(input_path):
        #List files to help debug Silva terminal
        print(f"Error: {input_path} not found. Files present: {os.listdir('.')}")
        sys.exit(1) # Exit the script with error code 1

    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(input_path)

    # Remove whitespace and ensure uppercase
    df['sequence'] = df['sequence'].str.strip().str.upper()

    # Print out the number of sequences
    print(f"Processing {len(df)} sequences...")

    # Calculate the biophysical descriptors using peptides.Peptide class
    # Define a function to get the metrics for each sequence
    def get_metrics(seq):
        # Create a Peptide object
        p = peptides.Peptide(seq)
        return pd.Series({
            'hydrophobicity': p.hydrophobicity(), # Hydrophobicity score - Kyte-Doolittle method
            'charge': p.charge(pH=7.4), #Charge score - Dexter-Moore's method
            'instability_index': p.instability_index(), #Safety anchor - higher than 40 is unstable - Guruprasad et. al. 1990
            'isoelectric_point': p.isoelectric_point() #pH where peptide is neutral - Balaban et. al. 1985
        })

    # Apply the function to the sequence column to get the metrics
    metrics_df = df['sequence'].apply(get_metrics)

    # Concatenate the original dataframe with the metrics dataframe
    final_df = pd.concat([df, metrics_df], axis=1)

    # Save the final dataframe to the output path
    final_df.to_csv(output_path, index=False)

    # Print out the resulting number of sequences in the final dataframe and the path
    print(f"Final dataframe with {len(final_df)} sequences and features saved to {output_path}.")

if __name__ == "__main__":
    # Check if Silva passed the JSON configuration as a command-line argument, otherwise look for env var directly
    config_json = sys.argv[1] if len(sys.argv) > 1 else os.getenv('DATA_PATHS')
    if config_json:
        try:
            config = json.loads(config_json)
            in_file = config.get('input', 'inputs/sequences.csv')
            out_file = config.get('output', 'outputs/ingested_sequences.csv')
            process_sequences(in_file, out_file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON configuration: {e}")
            print(f"Raw input received: {config_json}")
            sys.exit(1)
    else:
        print("Error: No JSON configuration provided.")
        sys.exit(1)