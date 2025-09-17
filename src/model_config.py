import os
import openai
import anthropic
import time
from openai import OpenAI

def prompt_model(model_name, prompt, temperature=0.8):
    print(f"Using model: {model_name}; temperature: {temperature}")
    if "claude" in model_name.lower():
        # time.sleep(20)  # avoid rate limiting
        client = anthropic.Anthropic(
            api_key=os.environ["CLAUDE_API_KEY"],
        )

        response = client.messages.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=temperature,
        )
        print(response)
        return response.content[0].text
    else:
        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            # base_url=os.environ["MODEL_SERVING_URL"],
        )

        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        print(response)
    return response.choices[0].message.content.strip()
