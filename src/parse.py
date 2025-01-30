from Bio import SeqIO
import pandas as pd

# Input and output file paths
input_fasta = "data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
output_csv = "data/processed/sequences.csv"

# Initialize lists to store parsed data
sequences = []
classes = []

# Parse FASTA file
for record in SeqIO.parse(input_fasta, "fasta"):
    # Extract structural class from header (e.g., a.1.1.1 -> 'a')
    header_parts = record.description.split()
    class_label = header_parts[1].split('.')[0]  # First letter of classification (a, b, c, etc.)
    
    # Append sequence and class to the lists
    sequences.append(str(record.seq))
    classes.append(class_label)

# Save to a DataFrame
df = pd.DataFrame({'sequence': sequences, 'class': classes})

# Save as a CSV file
df.to_csv(output_csv, index=False)
print(f"Data saved to {output_csv}")
