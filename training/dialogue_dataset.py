import json

# Sample dialogues
dialogues = [
    {"prompt": "Hello, how are you?", "response": "I'm doing well, thank you! How about you?"},
    {"prompt": "What is your name?", "response": "I'm an AI model here to help you."},
    {"prompt": "Tell me a joke.", "response": "Why don't scientists trust atoms? Because they make up everything!"},
    {"prompt": "What is AI?", "response": "AI stands for Artificial Intelligence, a field of computer science."},
    {"prompt": "Can you help me write a story?", "response": "Of course! What would you like the story to be about?"}
]

# Print the structure of dialogues
print(dialogues)

# Save to dialogue_dataset.json
try:
    with open('dialogue_dataset.json', 'w') as f:
        json.dump(dialogues, f, indent=4)
    print("dialogue_dataset.json created.")
except Exception as e:
    print(f"Error: {e}")
