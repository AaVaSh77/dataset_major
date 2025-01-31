# # Complex Architecture

# import torch
# import torch.nn as nn
# from torchviz import make_dot

# # Define the necessary building blocks
# class GatedBlock(nn.Module):
#     def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
#         super(GatedBlock, self).__init__()
#         conv = {(False, 1): nn.Conv1d,
#                 (True, 1): nn.ConvTranspose1d,
#                 (False, 2): nn.Conv2d,
#                 (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
#         self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
#         self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

#     def forward(self, x):
#         x1 = self.conv(x)
#         x2 = torch.sigmoid(self.gate(x))
#         out = x1 * x2
#         return out

# # Encoder class
# class Encoder(nn.Module):
#     def __init__(self, conv_dim=1, block_type='normal', n_layers=3):
#         super(Encoder, self).__init__()
#         block = {'normal': GatedBlock}[block_type]

#         layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]
#         for _ in range(n_layers - 1):
#             layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.main(x)

# # Carrier Decoder class
# class CarrierDecoder(nn.Module):
#     def __init__(self, conv_dim, block_type='normal', n_layers=4):
#         super(CarrierDecoder, self).__init__()
#         block = {'normal': GatedBlock}[block_type]

#         layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]
#         for _ in range(n_layers - 2):
#             layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))
#         layers.append(block(c_in=64, c_out=1, kernel_size=1, stride=1, padding=0, deconv=False))

#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.main(x)

# # Message Decoder class
# class MsgDecoder(nn.Module):
#     def __init__(self, conv_dim=1, block_type='normal'):
#         super(MsgDecoder, self).__init__()
#         block = {'normal': GatedBlock}[block_type]

#         self.main = nn.Sequential(
#             block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
#             block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
#             block(c_in=64, c_out=1, kernel_size=3, stride=1, padding=1, deconv=False)
#         )

#     def forward(self, x):
#         return self.main(x)

# # Audio Steganography Model (Without Discriminator)
# class AudioSteganographyModel(nn.Module):
#     def __init__(self, encoder_block_type='normal', decoder_block_type='normal', n_layers=3):
#         super(AudioSteganographyModel, self).__init__()

#         self.encoder = Encoder(conv_dim=1, block_type=encoder_block_type, n_layers=n_layers)
#         self.carrier_decoder = CarrierDecoder(conv_dim=64, block_type=decoder_block_type, n_layers=n_layers + 1)
#         self.message_decoder = MsgDecoder(conv_dim=64, block_type=decoder_block_type)

#     def forward(self, audio, message):
#         encoded_features = self.encoder(audio)
#         carrier_output = self.carrier_decoder(encoded_features)
#         message_output = self.message_decoder(encoded_features)
#         return carrier_output, message_output


# # Instantiate the model
# model = AudioSteganographyModel()

# # Dummy inputs for visualization
# audio_input = torch.randn(1, 1, 64, 64)  # Example spectrogram
# message_input = torch.randn(1, 1, 64, 64)

# # Forward pass
# carrier_output, message_output = model(audio_input, message_input)

# # Visualize the model
# graph = make_dot((carrier_output, message_output), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# # Save the graph to a file
# graph.render("audio_steganography_model_no_discriminator", format="png", cleanup=True)





































# # Simple Architecture

# import torch
# import torch.nn as nn
# from torchviz import make_dot

# # Simplified Gated Block
# class GatedBlock(nn.Module):
#     def __init__(self, c_in, c_out, kernel_size, stride, padding):
#         super(GatedBlock, self).__init__()
#         self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
#         self.gate = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)

#     def forward(self, x):
#         x1 = self.conv(x)
#         x2 = torch.sigmoid(self.gate(x))
#         return x1 * x2

# # Encoder
# class Encoder(nn.Module):
#     def __init__(self, c_in, c_out, n_layers):
#         super(Encoder, self).__init__()
#         layers = [GatedBlock(c_in, c_out, kernel_size=3, stride=1, padding=1)]
#         layers += [GatedBlock(c_out, c_out, kernel_size=3, stride=1, padding=1) for _ in range(n_layers - 1)]
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

# # Decoder
# class Decoder(nn.Module):
#     def __init__(self, c_in, c_out, n_layers):
#         super(Decoder, self).__init__()
#         layers = [GatedBlock(c_in, c_in, kernel_size=3, stride=1, padding=1) for _ in range(n_layers - 1)]
#         layers.append(GatedBlock(c_in, c_out, kernel_size=3, stride=1, padding=1))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

# # Steganography Model
# class AudioSteganographyModel(nn.Module):
#     def __init__(self):
#         super(AudioSteganographyModel, self).__init__()
#         self.encoder = Encoder(c_in=1, c_out=64, n_layers=3)
#         self.carrier_decoder = Decoder(c_in=64, c_out=1, n_layers=4)
#         self.message_decoder = Decoder(c_in=64, c_out=1, n_layers=3)

#     def forward(self, audio, message):
#         encoded = self.encoder(audio)
#         carrier = self.carrier_decoder(encoded)
#         secret = self.message_decoder(encoded)
#         return carrier, secret

# # Instantiate and visualize
# model = AudioSteganographyModel()
# audio_input = torch.randn(1, 1, 64, 64)
# message_input = torch.randn(1, 1, 64, 64)
# carrier, secret = model(audio_input, message_input)

# graph = make_dot((carrier, secret), params=dict(model.named_parameters()))
# graph.render("audio_model_architecture", format="png", cleanup=True)



















# # Very simple Architecture

# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyBboxPatch, Arrow

# # Function to create a block
# def create_block(ax, x, y, width, height, text, color):
#     block = FancyBboxPatch(
#         (x, y),
#         width,
#         height,
#         boxstyle="round,pad=0.1",
#         edgecolor="black",
#         facecolor=color,
#         lw=1.5,
#     )
#     ax.add_patch(block)
#     ax.text(
#         x + width / 2,
#         y + height / 2,
#         text,
#         ha="center",
#         va="center",
#         fontsize=10,
#         color="black",
#     )

# # Function to add arrows
# def add_arrow(ax, x_start, y_start, x_end, y_end):
#     arrow = Arrow(
#         x_start, y_start, x_end - x_start, y_end - y_start, width=0.1, color="black"
#     )
#     ax.add_patch(arrow)

# # Plotting the architecture
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.axis("off")

# # Encoder Block
# create_block(ax, 1, 6, 2, 1, "Encoder\n(3 Gated Blocks)", "#99CCFF")

# # Carrier Decoder
# create_block(ax, 4, 8, 2, 1, "Carrier Decoder\n(4 Gated Blocks)", "#FFCC99")

# # Message Decoder
# create_block(ax, 4, 4, 2, 1, "Message Decoder\n(3 Gated Blocks)", "#FFCC99")

# # Encoded Features
# create_block(ax, 3, 6.5, 1, 0.5, "Encoded\nFeatures", "#D9D9D9")

# # Outputs
# create_block(ax, 7, 8, 2, 1, "Carrier\nOutput", "#D9EAD3")
# create_block(ax, 7, 4, 2, 1, "Secret\nOutput", "#D9EAD3")

# # Arrows connecting blocks
# add_arrow(ax, 3, 7, 4, 8.5)  # To Carrier Decoder
# add_arrow(ax, 3, 6, 4, 4.5)  # To Message Decoder
# add_arrow(ax, 6, 8.5, 7, 8.5)  # Carrier Decoder to Output
# add_arrow(ax, 6, 4.5, 7, 4.5)  # Message Decoder to Output

# # Input blocks
# create_block(ax, 0, 7, 1, 1, "Audio\nInput", "#FFFF99")
# create_block(ax, 0, 5, 1, 1, "Message\nInput", "#FFFF99")
# add_arrow(ax, 1, 7.5, 2, 7.5)  # To Encoder
# add_arrow(ax, 1, 5.5, 2, 6.5)  # To Encoder

# # Display the diagram
# plt.show()













# # Simple convolution only architecture of the complex model 
# import torch
# import torch.nn as nn
# from torchviz import make_dot

# # Simplified Block Class
# class SimpleBlock(nn.Module):
#     def __init__(self, c_in, c_out, kernel_size, stride, padding):
#         super(SimpleBlock, self).__init__()
#         self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         return self.activation(self.conv(x))

# # Simplified Encoder
# class SimpleEncoder(nn.Module):
#     def __init__(self, c_in=1, c_out=64, n_layers=3):
#         super(SimpleEncoder, self).__init__()
#         layers = [SimpleBlock(c_in, c_out, kernel_size=3, stride=1, padding=1)]
#         for _ in range(n_layers - 1):
#             layers.append(SimpleBlock(c_out, c_out, kernel_size=3, stride=1, padding=1))
#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.main(x)

# # Simplified Decoder
# class SimpleDecoder(nn.Module):
#     def __init__(self, c_in=64, c_out=1, n_layers=3):
#         super(SimpleDecoder, self).__init__()
#         layers = [SimpleBlock(c_in, c_out, kernel_size=3, stride=1, padding=1)]
#         for _ in range(n_layers - 1):
#             layers.append(SimpleBlock(c_out, c_out, kernel_size=3, stride=1, padding=1))
#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.main(x)

# # Simplified Audio Steganography Model
# class SimpleAudioSteganographyModel(nn.Module):
#     def __init__(self, n_layers=3):
#         super(SimpleAudioSteganographyModel, self).__init__()
#         self.encoder = SimpleEncoder(c_in=1, c_out=64, n_layers=n_layers)
#         self.decoder = SimpleDecoder(c_in=64, c_out=1, n_layers=n_layers)

#     def forward(self, audio):
#         encoded = self.encoder(audio)
#         decoded = self.decoder(encoded)
#         return decoded

# # Instantiate the model
# model = SimpleAudioSteganographyModel()

# # Dummy input
# audio_input = torch.randn(1, 1, 64, 64)  # Example input

# # Forward pass
# output = model(audio_input)

# # Visualize the model
# graph = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# # Save the graph to a file
# graph.render("simple_audio_steganography_model", format="png", cleanup=True)



















# # For now FINAL MODELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL

# import torch
# import torch.nn as nn
# from torchviz import make_dot

# # Block
# class Block(nn.Module):
#     def __init__(self, c_in, c_out, kernel_size, stride, padding):
#         super(Block, self).__init__()
#         self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         return self.activation(self.conv(x))

# # Encode
# class Encode(nn.Module):
#     def __init__(self, c_in=1, c_out=64, n_layers=3):
#         super(Encode, self).__init__()
#         layers = [Block(c_in, c_out, kernel_size=3, stride=1, padding=1)]
#         for _ in range(n_layers - 1):
#             layers.append(Block(c_out, c_out, kernel_size=3, stride=1, padding=1))
#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.main(x)

# # Decode
# class Decode(nn.Module):
#     def __init__(self, c_in=64, c_out=1, n_layers=3):
#         super(Decode, self).__init__()
#         layers = [Block(c_in, c_out, kernel_size=3, stride=1, padding=1)]
#         for _ in range(n_layers - 1):
#             layers.append(Block(c_out, c_out, kernel_size=3, stride=1, padding=1))
#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.main(x)

# # Model
# class Model(nn.Module):
#     def __init__(self, n_layers=3):
#         super(Model, self).__init__()
#         self.encoder = Encode(c_in=1, c_out=64, n_layers=n_layers)
#         self.decoder = Decode(c_in=64, c_out=1, n_layers=n_layers)

#     def forward(self, audio):
#         encoded = self.encoder(audio)
#         decoded = self.decoder(encoded)
#         return decoded

# # Instantiate the model
# model = Model()

# # Dummy input
# audio_input = torch.randn(1, 1, 64, 64)  # Example input

# # Forward pass
# output = model(audio_input)

# # Visualize the model
# graph = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=True)

# # Save the graph to a file
# graph.render("audio_model_simplified_names", format="png", cleanup=True)

















import torch
import torch.nn as nn
from torchviz import make_dot

# Block
class Block(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))

# Secret Encoding Block
class SecretEncoding(nn.Module):
    def __init__(self, c_in=1, c_out=64, n_layers=3):
        super(SecretEncoding, self).__init__()
        layers = [Block(c_in, c_out, kernel_size=3, stride=1, padding=1)]
        for _ in range(n_layers - 1):
            layers.append(Block(c_out, c_out, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Concealment Block
class Concealment(nn.Module):
    def __init__(self, c_in=64, c_out=64, n_layers=3):
        super(Concealment, self).__init__()
        layers = [Block(c_in, c_out, kernel_size=3, stride=1, padding=1)]
        for _ in range(n_layers - 1):
            layers.append(Block(c_out, c_out, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Decoding Block
class Decoding(nn.Module):
    def __init__(self, c_in=64, c_out=1, n_layers=3):
        super(Decoding, self).__init__()
        layers = [Block(c_in, c_out, kernel_size=3, stride=1, padding=1)]
        for _ in range(n_layers - 1):
            layers.append(Block(c_out, c_out, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Model
class Model(nn.Module):
    def __init__(self, n_layers=3):
        super(Model, self).__init__()
        self.secret_encoding = SecretEncoding(c_in=1, c_out=64, n_layers=n_layers)
        self.concealment = Concealment(c_in=64, c_out=64, n_layers=n_layers)
        self.decoding = Decoding(c_in=64, c_out=1, n_layers=n_layers)

    def forward(self, audio):
        encoded = self.secret_encoding(audio)
        concealed = self.concealment(encoded)
        decoded = self.decoding(concealed)
        return decoded

# Instantiate the model
model = Model()

# Dummy input
audio_input = torch.randn(1, 1, 64, 64)  # Example input

# Forward pass
output = model(audio_input)

# Visualize the model
graph = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# Save the graph to a file
graph.render("audio_model_with_superblocks", format="png", cleanup=True)