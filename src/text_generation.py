# This section of code will cover text generation task number 5.
# And we are going to use Open AI API's to comeplte this task
from config import client

def generate_text(prompt):
    response = client.responses.create(
        model = "gpt-4o-mini",
        input = prompt
    )
    return response.output_text
