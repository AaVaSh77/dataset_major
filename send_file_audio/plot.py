import re
import matplotlib.pyplot as plt

# Initialize lists to store data
epochs = []
carrier_loss = []
carrier_snr = []
message_loss = []
message_snr = []

# Read data from stdout.log
with open('stdout.log', 'r') as file:
    for line in file:
        # Match epoch information
        epoch_match = re.search(r"==> suffix: (\d+)_epoch", line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))

        # Match carrier loss
        carrier_loss_match = re.search(r"carrier loss: ([\d\.\-e]+)", line)
        if carrier_loss_match:
            carrier_loss.append(float(carrier_loss_match.group(1)))

        # Match carrier SNR
        carrier_snr_match = re.search(r"carrier SnR: ([\d\.\-e]+)", line)
        if carrier_snr_match:
            carrier_snr.append(float(carrier_snr_match.group(1)))

        # Match message loss
        message_loss_match = re.search(r"message loss: ([\d\.\-e]+)", line)
        if message_loss_match:
            message_loss.append(float(message_loss_match.group(1)))

        # Match message SNR
        message_snr_match = re.search(r"message SnR: ([\d\.\-e]+)", line)
        if message_snr_match:
            message_snr.append(float(message_snr_match.group(1)))

# Plotting
plt.figure(figsize=(12, 8))

# Carrier Loss vs Epochs
plt.subplot(2, 2, 1)
plt.plot(epochs, carrier_loss, marker='o', label='Carrier Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Carrier Loss vs Epochs')
plt.grid(True)
plt.legend()

# Carrier SNR vs Epochs
plt.subplot(2, 2, 2)
plt.plot(epochs, carrier_snr, marker='o', label='Carrier SNR', color='orange')
plt.xlabel('Epochs')
plt.ylabel('SNR')
plt.title('Carrier SNR vs Epochs')
plt.grid(True)
plt.legend()

# Message Loss vs Epochs
plt.subplot(2, 2, 3)
plt.plot(epochs, message_loss, marker='o', label='Message Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Message Loss vs Epochs')
plt.grid(True)
plt.legend()

# Message SNR vs Epochs
plt.subplot(2, 2, 4)
plt.plot(epochs, message_snr, marker='o', label='Message SNR', color='red')
plt.xlabel('Epochs')
plt.ylabel('SNR')
plt.title('Message SNR vs Epochs')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
