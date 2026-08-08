# This section of code will cover text generation task number 5.
# And we are going to use Open AI API's to comeplte this task

from pyarrow import null


def generate_text(client, prompt, provider):
    if provider == "openai":
        response = client.responses.create(
            model = "gpt-4o-mini",
            input = prompt
        )
        return response.output_text
    elif provider == "hf":
        return null
    else:
        return null
