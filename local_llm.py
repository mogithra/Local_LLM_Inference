import pandas as pd
import ollama

# Load prompts
prompts = pd.read_csv("dataset/prompts.csv")

# Local model
model = "llama3.2"

responses = []

for index, row in prompts.iterrows():

    prompt = row["prompt"]

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print("Prompt:", prompt)
    print("Response:", answer)
    print("-" * 50)

    responses.append({
        "prompt": prompt,
        "response": answer
    })

# Save responses
results = pd.DataFrame(responses)

results.to_csv(
    "dataset/llm_responses.csv",
    index=False
)

print("Responses saved successfully!")
