import torchaudio
import pesto

# Load audio (ensure mono; stereo channels are treated as separate batch dimensions)
file_path = '/home/ysm/Audio_Feature_Extractor/data_for_testing/prompt.wav'
x, sr = torchaudio.load(file_path)
x = x.mean(dim=0)  # PESTO takes mono audio as input

# Predict pitch. x can be (num_samples) or (batch, num_samples)
timesteps, pitch, confidence, activations = pesto.predict(x, sr)

for i, res in enumerate(zip(timesteps, pitch)):
    print(f'{i} >>> timesteps: {res[0]}       pitch: {res[1]}')

# Using a custom checkpoint:
# predictions = pesto.predict(x, sr, model_name="/path/to/checkpoint.ckpt")

# Predicting from multiple files:
# pesto.predict_from_files(["example1.wav", "example2.mp3"], export_format=["csv"])