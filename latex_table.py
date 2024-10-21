import pandas as pd

# Load the Excel data
data = pd.read_excel('ANVNSimultaneous_1n_root_bank_thresh1.xlsx')
threshs = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30, 35, 40, 45, 50]

# Function to create a LaTeX table with combined accuracy and average spikes
def create_latex_table(data, colored_func_name_acc):
    # Pivot the data to get Max Energy as columns and Energy as rows
    pivoted_data = data.pivot(index='Energy', columns='Max Energy')

    num_max_energy = len(pivoted_data.index)

    # Start constructing the LaTeX table
    table = "\\begin{center}\n"
    table += "\\begin{tabular}{|c|" + "|".join(["c"] * num_max_energy) + "|}\n"  
    table += "\\hline\n"
    
    # Create header row for Max Energy
    header = "Energy & " + " & ".join(map(str, threshs[6:num_max_energy+6])) + " \\\\\n"
    table += header
    table += "\\hline\n"
    
    # Iterate through each energy level to construct rows
    for energy in pivoted_data.index:
        # Extracting accuracy and average spikes for the given energy
        accuracy_row = pivoted_data.loc[energy, 'SNN Accuracy']
        # accuracy_row = accuracy_row[6::]
        spikes_row = pivoted_data.loc[energy, 'Average Spikes']
        # spikes_row = spikes_row[6::]
        
        # Create entries for accuracy and average spikes
        entries = " & ".join([f"\\{colored_func_name_acc}{{{accuracy:.2f}}}{{{int(round(spike))}}}" 
                              for accuracy, spike in zip(accuracy_row, spikes_row)])
        
        table += f"{energy} & {entries} \\\\\n"
        table += "\\hline\n"
    
    table += "\\end{tabular}\n"
    table += "\\end{center}\n"
    return table



# Create the combined table
acc = create_latex_table(data, 'coloredCellAcc')
sp = create_latex_table(data, 'coloredCellASPM')

# Print the table
print("Combined Accuracy and Average Spikes Table:")
print(acc)
print(sp)
