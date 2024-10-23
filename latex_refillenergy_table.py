import pandas as pd

# Load the Excel data
data = pd.read_excel('ANVNSimultaneous_1n_root_bank_varyenergy.xlsx')
# threshs = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30, 35, 40, 45, 50]
threshs = data['Max Energy'].unique()

# Function to create a LaTeX table with combined accuracy and average spikes
def create_latex_table(data, colored_func_name_acc, re):
    # Pivot the data to get Max Energy as columns and Energy as rows
    pivoted_data = data.pivot(index='Energy', columns='Max Energy')

    num_max_energy = len(pivoted_data.index)

    # Start constructing the LaTeX table
    table = "\\begin{table}\n"
    table += "\\begin{center}\n"
    table += "\\begin{tabular}{|c|" + "|".join(["c"] * num_max_energy) + "|}\n"  
    table += "\\hline\n"
    
    # Create header row for Max Energy
    header = "Energy & " + " & ".join(map(str, threshs)) + " \\\\\n"
    table += header
    table += "\\hline\n"
    
    # Iterate through each energy level to construct rows
    for energy in pivoted_data.index:
        # Extracting accuracy and average spikes for the given energy
        accuracy_row = pivoted_data.loc[energy, 'SNN Accuracy']
        spikes_row = pivoted_data.loc[energy, 'Average Spikes']
        
        # Create entries for accuracy and average spikes
        entries = " & ".join([f"\\{colored_func_name_acc}{{{accuracy:.2f}}}{{{int(round(spike))}}}" 
                              for accuracy, spike in zip(accuracy_row, spikes_row)])
        
        table += f"{energy} & {entries} \\\\\n"
        table += "\\hline\n"
    
    table += "\\end{tabular}\n"
    table += "\\end{center}\n"
    table += "\caption{Accuracy of SNN with ANVN Simultaneously Trained (converted), Augmented Root Bank, and Thresholds Adjusted. Flows "+str(re)+" Energy Every $n=1$ Timesteps}\n"
    table += "\label{table:anvnsim_rootbank_vary_energy_1n_" + str(re) + "re}\n"
    table += "\end{table}\n"
    return table

# Loop through unique refill energies
unique_refill_energies = data['Refill Energy'].unique()

combined_tables = []

for refill_energy in unique_refill_energies:
    # Filter data for the current refill energy
    filtered_data = data[data['Refill Energy'] == refill_energy]
    
    # Generate LaTeX tables for accuracy and spikes
    acc_table = create_latex_table(filtered_data, 'coloredCellAcc', refill_energy)
    spikes_table = create_latex_table(filtered_data, 'coloredCellASPM', refill_energy)
    
    # Add a section in the LaTeX for this refill energy
    combined_tables.append(f"\\section*{{Refill Energy = {refill_energy}}}\n")
    combined_tables.append(acc_table)
    combined_tables.append(spikes_table)

# Combine all the tables for each refill energy
final_latex_document = "\n".join(combined_tables)

# Print or save the final LaTeX document
print("Final LaTeX Document with Refill Energy:")
print(final_latex_document)
