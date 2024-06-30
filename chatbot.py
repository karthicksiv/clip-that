import os
import requests
import openai

# Set your API key here
API_KEY = "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"


# Define the API endpoint
API_URL = "https://api.perplexity.ai/v1/chat/completions"

# Initialize the OpenAI client
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

# Function to get a response from the Perplexity API
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

# Main function to run the chatbot
def main():
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