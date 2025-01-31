import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'modified.csv'  # Update the path if needed
data = pd.read_csv(csv_file_path)

# Group training and validation metrics
metrics_pairs = [
    ("train_carrier_loss", "val_carrier_loss"),
    ("train_message_loss", "val_message_loss"),
    ("train_carrier_snr", "val_carrier_snr"),
    ("train_msg_snr", "val_msg_snr"),
    ("train_accuracy", "val_accuracy")
]

# Create a subplot with 3 rows and 2 columns
fig = plt.figure(figsize=(12, 12))  # Adjust figure size as needed

# First subplot (1)
plt.subplot(3, 3, 1)
plt.plot(data['epoch'], data['train_carrier_loss'], label='train_carrier_loss', color='blue', linestyle='-', linewidth=1)
plt.plot(data['epoch'], data['val_carrier_loss'], label='val_carrier_loss', color='red', linestyle='--', linewidth=1)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Carrier Loss vs Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Second subplot (2)
plt.subplot(3, 3, 2)
plt.plot(data['epoch'], data['train_message_loss'], label='train_message_loss', color='blue', linestyle='-', linewidth=1)
plt.plot(data['epoch'], data['val_message_loss'], label='val_message_loss', color='red', linestyle='--', linewidth=1)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Message Loss vs Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Third subplot (3)
plt.subplot(3, 3, 3)
plt.plot(data['epoch'], data['train_carrier_snr'], label='train_carrier_snr', color='blue', linestyle='-', linewidth=1)
plt.plot(data['epoch'], data['val_carrier_snr'], label='val_carrier_snr', color='red', linestyle='--', linewidth=1)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Carrier SNR vs Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Fourth subplot (4)
plt.subplot(3, 3, 4)
plt.plot(data['epoch'], data['train_msg_snr'], label='train_msg_snr', color='blue', linestyle='-', linewidth=1)
plt.plot(data['epoch'], data['val_msg_snr'], label='val_msg_snr', color='red', linestyle='--', linewidth=1)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Message SNR vs Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Fifth subplot (5)
plt.subplot(3, 3, 5)
plt.plot(data['epoch'], data['train_accuracy'], label='train_accuracy', color='blue', linestyle='-', linewidth=1)
plt.plot(data['epoch'], data['val_accuracy'], label='val_accuracy', color='red', linestyle='--', linewidth=1)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Accuracy vs Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent overlapping labels and titles
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase both vertical and horizontal space

# Save the entire figure as a single image
plt.savefig('all_metrics_vs_epochs.png')  # Save the combined plot as an image

# Show the combined plot
plt.show()

# Save individual plots for each pair
for train_col, val_col in metrics_pairs:
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data[train_col], label=train_col, color='blue', linestyle='-', linewidth=1)
    plt.plot(data['epoch'], data[val_col], label=val_col, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(f'{train_col} and {val_col} vs Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{train_col}_and_{val_col}_vs_epochs.png')  # Save the plot as an image
    plt.close()
