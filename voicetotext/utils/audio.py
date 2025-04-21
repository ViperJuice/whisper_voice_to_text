import numpy as np

def is_silent(audio_data, silence_threshold=1000, dtype=np.int16):
    """
    Determines if the audio data contains silence based on a threshold.
    
    Args:
        audio_data: The audio data to analyze
        silence_threshold: The amplitude threshold below which is considered silence
        dtype: The numpy data type of the audio data
        
    Returns:
        bool: True if the audio is silent, False otherwise
    """
    if len(audio_data) == 0:
        return True
        
    # Convert to numpy array if needed
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.frombuffer(audio_data, dtype=dtype)
    
    # Calculate the average amplitude of the audio
    amplitude = np.abs(audio_data).mean()
    
    # Return True if below threshold
    return amplitude < silence_threshold 