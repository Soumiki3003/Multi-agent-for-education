from google import genai

client = genai.Client(api_key="AIzaSyDchhG7QSTBD0qnHWmVzcUh5sIOAMUslBo")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Are bunnies considered cute? Also what is your token limit? I got pro.",
)
print(response.text)
