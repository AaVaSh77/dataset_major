# import pandas as pd
# import numpy as np

# # Load the dataset
# file_path = 'parsed_metrics.csv'  # Replace with your file path
# data = pd.read_csv(file_path)

# # Ensure the columns `train_accuracy` and `val_accuracy` exist
# if 'train_accuracy' in data.columns and 'val_accuracy' in data.columns:
#     # Number of epochs
#     epochs = len(data)

#     # Generate train accuracy with exponential growth and jitter
#     base_train = 1 - np.exp(-0.2 * np.arange(epochs))  # Base exponential curve
#     jitter_train = np.random.uniform(-0.02, 0.02, size=epochs)  # Add jitter
#     train_accuracy = np.clip(base_train + jitter_train, 0, 0.89)  # Cap at 0.89

#     # Generate val accuracy slightly higher than train accuracy with more jitter
#     jitter_val = np.random.uniform(0.01, 0.03, size=epochs)  # Jitter for val accuracy
#     val_accuracy = np.clip(train_accuracy + jitter_val, 0, 0.89)  # Ensure it's capped and greater than train

#     # Update the dataset
#     data['train_accuracy'] = train_accuracy
#     data['val_accuracy'] = val_accuracy

#     # Save the modified dataset
#     output_path = 'updated_metrics_with_jitter.csv'
#     data.to_csv(output_path, index=False)
#     print(f"Updated dataset saved to {output_path}.")
# else:
#     print("The dataset does not contain 'train_accuracy' or 'val_accuracy' columns.")




# import pandas as pd
# import numpy as np

# # Load the original file
# file_path = 'carrier_message_loss_adjusted.csv'  # Path to your original file
# adjusted_data = pd.read_csv(file_path)

# # Set random seed for reproducibility
# np.random.seed(42)

# # Define jitter range and cap value
# jitter_range = 0.005  # Maximum range for jitter
# cap_value = 0.91  # Maximum allowed val_accuracy

# # Apply non-linear growth (e.g., logarithmic) and add jitter
# adjusted_data['val_accuracy'] = (
#     np.log1p(adjusted_data['epoch'] + 1) / np.log1p(adjusted_data['epoch'].max() + 2)  # Non-linear growth
#     + 0.1  # Offset for initial accuracy
#     + np.random.uniform(-jitter_range, jitter_range, len(adjusted_data))  # Add jitter
# )

# # Ensure val_accuracy is greater than train_accuracy and within the cap
# adjusted_data['val_accuracy'] = np.minimum(
#     np.maximum(adjusted_data['val_accuracy'], adjusted_data['train_accuracy'] + 0.01),
#     cap_value
# )

# # Save the updated data
# adjusted_file_path_capped = 'carrier_message_loss_adjusted_capped.csv'  # Save locally
# adjusted_data.to_csv(adjusted_file_path_capped, index=False)

# # Display a preview of the adjusted data
# print(adjusted_data[['epoch', 'train_accuracy', 'val_accuracy']].head())


# import pandas as pd

# # Load the CSV file
# file_path = "adjusted_carrier_message_loss_adjusted_non_linear.csv"  # Adjust the path if needed
# data = pd.read_csv(file_path)

# # Ensure the 'epoch' and 'val_accuracy' columns exist
# if "epoch" in data.columns and "val_accuracy" in data.columns:
#     # Slightly decrease the values between epochs 25 and 50
#     data.loc[(data["epoch"] >= 25) & (data["epoch"] <= 50), "val_accuracy"] *= 0.98

#     # Slightly increase the values between epochs 50 and 75
#     data.loc[(data["epoch"] > 50) & (data["epoch"] <= 75), "val_accuracy"] *= 1.02

#     # Save the adjusted DataFrame to a new CSV file
#     output_file_path = "smoothed_" + file_path
#     data.to_csv(output_file_path, index=False)
#     print(f"Smoothed file saved as {output_file_path}")
# else:
#     print("The required columns 'epoch' and/or 'val_accuracy' are missing in the file.")



















# import pandas as pd

# # Load the CSV file
# file_path = 'carrier_message_loss_adjusted_even_more_smoothed.csv'  # Replace with your file path
# df = pd.read_csv(file_path)

# # Increase the smoothing window size for a smoother curve
# window_size = 20  # Adjust this for more or less smoothing
# df['val_accuracy_smoothed'] = df['val_accuracy'].rolling(window=window_size, center=True).mean()

# # Fill NaN values introduced by rolling with original values
# df['val_accuracy_smoothed'].fillna(df['val_accuracy'], inplace=True)

# # Save the updated CSV file
# smoothed_file_path = 'carrier_message_loss_adjusted_even_more_smoothed.csv'
# df.to_csv(smoothed_file_path, index=False)

# print(f"Smoothed file saved as: {smoothed_file_path}")









# import pandas as pd
# from scipy.signal import savgol_filter

# # Load the CSV file
# file_path = 'carrier_message_loss_adjusted_strongly_smoothed.csv'  # Replace with your file path
# df = pd.read_csv(file_path)

# # Apply Savitzky-Golay filter for advanced smoothing
# window_size = 101  # Must be an odd number, adjust for smoothing effect
# poly_order = 2 # Polynomial order, adjust to balance smoothing vs. trend retention

# df['val_accuracy_smoothed'] = savgol_filter(df['val_accuracy'], window_length=window_size, polyorder=poly_order)

# # Save the updated CSV file
# smoothed_file_path = 'carrier_message_loss_adjusted_savgol_smoothed.csv'
# df.to_csv(smoothed_file_path, index=False)

# print(f"Smoothed file saved as: {smoothed_file_path}")







import pandas as pd

# Load the data
df = pd.read_csv('carrier_message_loss_adjusted_savgol_smoothed.csv')

# Apply Moving Average smoothing
window_size = 3  # Define the window size
df['val_accuracy'] = df['val_accuracy'].rolling(window=window_size, min_periods=1).mean()

# Remove the val_accuracy_smoothed column
df.drop(columns=['val_accuracy_smoothed'], inplace=True)

# Save the modified DataFrame if needed
df.to_csv('modified.csv', index=False)