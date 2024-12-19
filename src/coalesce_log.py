import datetime
import matplotlib.pyplot as plt


# Function to read data from a file and extract lines where the last part changes
def extract_changed_lines_from_file(file_path):
    result = []
    last_file = None
    last_line = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Extract the last "word" (filename or path) in the line
        current_file = line.split()[-1]

        # If it changes from the previous, add the line to the result
        if current_file != last_file:
            if last_line is not None:
                result.append(last_line.strip())
            result.append(line.strip())
            last_file = current_file

        last_line = line

    # Always include the last line
    if lines[-1].strip() not in result:
        result.append(lines[-1].strip())

    return result


def parse_data_for_plotting(lines):
    data = []

    # Calculate the base timestamp (absolute to relative conversion)
    first_timestamp = datetime.datetime.strptime(lines[0].strip().split()[0][:-3], "%H:%M:%S.%f")

    for i in range(0, len(lines), 2):
        # Odd line for start info
        start_line = lines[i].strip()
        start_parts = start_line.split()
        start_timestamp = datetime.datetime.strptime(start_parts[0][:-3], "%H:%M:%S.%f")  # Remove last 3 digits
        start_relative_ms = int((start_timestamp - first_timestamp).total_seconds() * 1000)

        # Even line for end info
        end_line = lines[i + 1].strip()
        end_parts = end_line.split()
        end_timestamp = datetime.datetime.strptime(end_parts[0][:-3], "%H:%M:%S.%f")  # Remove last 3 digits
        end_relative_ms = int((end_timestamp - first_timestamp).total_seconds() * 1000)

        # Extract the size (pivot around '[fd=3  ]')
        size_index = end_parts.index("[fd=3") + 2  # The size is two entries after '[fd=3'
        size_bytes = int(end_parts[size_index])
        size_mb = size_bytes / (1024 * 1024)  # Convert to MB

        # Extract the batch name
        batch_name = end_parts[-1].split("/")[-1]

        # Add the processed data
        data.append((batch_name, start_relative_ms, end_relative_ms, f"{size_mb:.1f}MB"))

    return data


def parse_data_for_offset_plotting(file_path):
    data = []
    last_batch_name = None

    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = lines[:192]

    for i in range(0, len(lines)):
        # Extract the batch name
        end_line = lines[i].strip()
        end_parts = end_line.split()
        batch_name = end_parts[-1].split("/")[-1]

        if batch_name != last_batch_name:
            # Calculate the base timestamp (absolute to relative conversion)
            first_timestamp = datetime.datetime.strptime(lines[i].strip().split()[0][:-3], "%H:%M:%S.%f")
            last_batch_name = batch_name

        # Even line for end info
        end_timestamp = datetime.datetime.strptime(end_parts[0][:-3], "%H:%M:%S.%f")  # Remove last 3 digits
        end_relative_ms = ((end_timestamp - first_timestamp).total_seconds() * 1000)

        # Extract the size (pivot around '[fd=3  ]')
        size_index = end_parts.index("[fd=3") + 2  # The size is two entries after '[fd=3'
        size_bytes = int(end_parts[size_index])
        size_mb = size_bytes / (1024 * 1024)  # Convert to MB



        # Add the processed data
        # data.append((float(f"{end_relative_ms:.1f}"), float(f"{size_mb:.1f}"), batch_name))
        # offset data
        data.append((float(f"{end_relative_ms:.1f}"), float(f"{size_bytes:.1f}"), batch_name))

    return data


def plot_culmulative_size(data):
    # Updated plot to overlay data by batch names

    # Extract unique batch names and assign light colors
    batch_names = sorted(set(entry[2] for entry in data))
    batch_colors = {batch: plt.cm.Pastel1(i / len(batch_names)) for i, batch in enumerate(batch_names)}

    # Plotting the cumulative data
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all data points, grouped by batches but in the same plot
    for batch in batch_names:
        batch_data = [(x, y) for x, y, b in data if b == batch]
        x_data, y_data = zip(*batch_data)
        ax.plot(x_data, y_data, marker='o', linestyle='-', label=batch, color=batch_colors[batch])

    # Formatting the plot
    ax.set_xlabel("Relative Time (ms)")
    ax.set_ylabel("Offset (Bytes)")
    ax.set_title("Read Offset Over Time (Overlaid)")
    ax.legend(title="Batch Names")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


###
#   The full code for the bar-plot needs to reach ChatGPT, below is the snippet from its analysis tool.
###
# # Plotting the filtered data with light colors for bars, black annotations, and colored y-axis labels
# fig, ax = plt.subplots(figsize=(12, 6))
#
# # Assigning distinct light colors for batches
# unique_batches = sorted(set(item[0] for item in filtered_data))
# batch_colors = {batch: plt.cm.Pastel1(i / len(unique_batches)) for i, batch in enumerate(unique_batches)}
#
# for batch_name, start, end, size in filtered_data:
#     if start != end:  # Skip zero-length bars
#         ax.barh(batch_name, end - start, left=start, color=batch_colors[batch_name])
#         ax.text((start + end) / 2, batch_name, size, ha="center", va="center", fontsize=8, color="black")
#
# # Coloring y-axis text
# ytick_labels = ax.get_yticklabels()
# for label, batch_name in zip(ytick_labels, unique_batches):
#     label.set_color(batch_colors[batch_name])
#
# # Formatting the plot
# ax.set_xlabel("Time (ms)")
# ax.set_ylabel("Data Batches")
# ax.set_xlim(0, max(item[2] for item in filtered_data) + 100)
# ax.set_yticks(range(len(unique_batches)))
# ax.set_yticklabels(unique_batches)
# ax.grid(axis="x", linestyle="--", alpha=0.5)
#
# plt.tight_layout()
# plt.show()



# Example usage: Replace 'your_file.txt' with the path to your file
lines = extract_changed_lines_from_file('profile/read.log')
data = parse_data_for_plotting(lines)
for d in data:
    print(d)

offset_data = parse_data_for_offset_plotting('profile/read.log')
for d in offset_data:
    print(d)
plot_culmulative_size(offset_data)