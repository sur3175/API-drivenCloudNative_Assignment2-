# This section of code will cover text generation task number 5.
# And we are using Open AI and Hugging face API's to comeplte this task

from logger import log

def generate_text(client, prompt, provider):

    log.info( f"Text generation started | provider={provider}" )

    if provider == "openai":
        model = "gpt-4o-mini"
        log.info( f"Calling OpenAI | model={model}" )

        try:
            response = client.responses.create(
                model=model,
                input=prompt
            )
        except Exception:
            log.exception(
                f"Text generation failed | "
                f"provider={provider} | model={model}"
            )
            raise

        if response.output_text:
            log.info( "OpenAI text generation completed successfully" )
        else:
            log.warning( "OpenAI returned an empty response" )

        return response.output_text

    elif provider == "hf":
        model = "openai/gpt-oss-120b:cheapest"
        log.info( f"Calling Hugging Face | model={model}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        except Exception:
            log.exception(
                f"Text generation failed | "
                f"provider={provider} | model={model}"
            )
            raise

        if response.choices and response.choices[0].message.content:
            log.info( "Hugging Face text generation completed successfully" )
        else:
            log.warning( "Hugging Face returned an empty response")

        return response.choices[0].message.content

    else:
        raise ValueError( f"Unsupported provider: {provider}")
