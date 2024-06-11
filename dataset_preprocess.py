import os
import shutil

# Define mapping for emotion labels
emotion_mapping = {
    '01': 'Neutral',
    '02': 'Calm',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Anger',
    '06': 'Fearful',
    '07': 'Disgust',
    '08': 'Surprised'
}

# Path to the RAVDESS dataset folder
dataset_folder = 'RAVDESS'

# Output folder for the organized dataset
output_folder = 'RAVDESS_Output'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through actor folders
for actor_folder in os.listdir(dataset_folder):
    actor_path = os.path.join(dataset_folder, actor_folder)
    if os.path.isdir(actor_path):
        # Iterate through audio files in actor folder
        for audio_file in os.listdir(actor_path):
            if audio_file.endswith('.wav'):
                # Parse filename to extract emotion label
                filename_parts = audio_file.split('-')
                emotion_label = emotion_mapping[filename_parts[2]]
                
                # Create emotion folder if it doesn't exist
                emotion_folder = os.path.join(output_folder, emotion_label)
                if not os.path.exists(emotion_folder):
                    os.makedirs(emotion_folder)
                
                # Move audio file to emotion folder
                src_file = os.path.join(actor_path, audio_file)
                dst_file = os.path.join(emotion_folder, audio_file)
                shutil.move(src_file, dst_file)

print("Dataset folder structure organized successfully.")
