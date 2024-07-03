import os
import requests
import openai
import pandas as pd
import csv
import json

# Set your API key here
API_KEY = "api"

# Define the API endpoint
API_URL = "https://api.perplexity.ai/v1/chat/completions"

# Initialize the OpenAI client
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

def convert_csv_to_jsonl(csv_file, jsonl_file):
    system_message = {
        "role": "system",
        "content": (
            "You are an AI assistant designed to help users find specific YouTube videos and timestamps based on their descriptions of clips or sound bytes. Your task is to identify the most relevant YouTube video and provide the exact timestamp where the described clip or sound byte occurs."
            "Users will describe a scene, dialogue, or sound byte they remember from a YouTube video. The descriptions may vary in detail and accuracy. Your goal is to use the description to search for the most relevant video and pinpoint the timestamp."
            "The user will provide a description of a clip or sound byte they saw or heard. For example, 'a scene where a cat jumps on a table and knocks over a vase.'"
            "Keywords: YouTube video, Timestamp, Clip description, Sound byte, Scene"
            "Respond with the YouTube video link and the exact timestamp in the following format:"
            "- Video: [YouTube Video URL]"
            "- Timestamp: [HH:MM:SS]"
        ),
    }

    with open(csv_file, 'r') as f_csv, open(jsonl_file, 'w') as f_jsonl:
        reader = csv.DictReader(f_csv)
        for row in reader:
            user_message = {"role": "user", "content": row["prompt"]}
            assistant_message = {
                "role": "assistant",
                "content": f"Here is a video link: {row['video_link']} at timestamp {row['timestamp']}"
            }
            conversation = {"messages": [system_message, user_message, assistant_message]}
            json.dump(conversation, f_jsonl)
            f_jsonl.write('\n')

def fine_tune_model_locally(jsonl_file):
    fine_tune_endpoint = 'https://api.perplexity.ai/v1/fine-tune'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    with open(jsonl_file, 'r') as f:
        data = {
            'training_data': f.read(),
            'model': 'llama-3-sonar-large-32k-online',
            'epochs': 3,
            'batch_size': 16
        }
    response = requests.post(fine_tune_endpoint, headers=headers, json=data)
    if response.status_code == 200:
        print('Fine-tuning started successfully')
    else:
        print(f'Error: {response.status_code} - {response.text}')

def get_response(prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant designed to help users find specific YouTube videos and timestamps based on their descriptions of clips or sound bytes. Your task is to identify the most relevant YouTube video and provide the exact timestamp where the described clip or sound byte occurs."
                "Users will describe a scene, dialogue, or sound byte they remember from a YouTube video. The descriptions may vary in detail and accuracy. Your goal is to use the description to search for the most relevant video and pinpoint the timestamp."
                "The user will provide a description of a clip or sound byte they saw or heard. For example, 'a scene where a cat jumps on a table and knocks over a vase.'"
                "Keywords: YouTube video, Timestamp, Clip description, Sound byte, Scene"
                "Respond with the YouTube video link and the exact timestamp in the following format:"
                "- Video: [YouTube Video URL]"
                "- Timestamp: [HH:MM:SS]"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3-sonar-large-32k-online",
            messages=messages,
        )
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Convert CSV to JSONL
    csv_file = 'training_set.csv'
    jsonl_file = 'training_set.jsonl'
    convert_csv_to_jsonl(csv_file, jsonl_file)

    # Fine-tune the model locally
    fine_tune_model_locally(jsonl_file)

    # Run the chatbot
    print("Say hello to your new assistant! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = get_response(user_input)
        if response:
            print("Assistant:", response.choices[0].message.content)
        else:
            print("Failed to get a valid response from the API.")

if __name__ == "__main__":
    main()