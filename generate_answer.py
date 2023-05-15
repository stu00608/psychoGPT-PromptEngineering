"""This script use openai gpt model to generate multiple dummy answers to a list of questions.
"""

import os
import re
import json
import time
import openai
import traceback
import tiktoken
import argparse
import threading
from tqdm import tqdm

responses = []
usages = []

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extract_answer(string):
    patterns = [r"RESPONSE\n(.*?)```markdown(.*?)```", r"```markdown(.*?)```", r"```(.*?)```"]

    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, string, re.DOTALL)
        if len(matches) == 1:
            if i == 0 and isinstance(matches[0], tuple):
                return matches[0][1].strip()
            else:
                return matches[0].strip()
        else:
            return None

    return None 

def make_prompt(questions: list, return_str: bool = False):
    questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    text = f"""The assistant must return the following structure:

RESPONSE
```markdown
1. answer1
2. answer2
...
14. answer14
```

questions:
{questions}

program:
- Randomly generate 5 human characters with different gender, age, jobs.
- From those characters, choose one that act as one seeking psychotherapy treatment.
- Make up a story that cause this character to seek psychotherapy treatment.
- Act as this character and answer questions above in Traditional Chinese. Return the tag RESPONSE just before your report. 

State each step of the program and show your work for performing that step.

1: Randomly generate 5 human characters with different gender, age, jobs.

"""

    if return_str:
        return text

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text}
    ]

    return prompt

def chat_request(prompt: list, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
    while True:
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError as e:
            print("Rate limit reached. Waiting 10 seconds and retry...")
            time.sleep(10)
        except openai.error.APIError as e:
            print("API error. Waiting 10 seconds and retry...")
            time.sleep(10)
        except Exception as e:
            print("Unknown error. Waiting 10 seconds and retry...")
            traceback.print_exc()
            return None, None

    response = completions['choices'][0]['message']['content']
    usage = (completions['usage']['prompt_tokens'], completions['usage']['completion_tokens'])
    return response, usage

def task(questions: list, progress_bar: tqdm, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
    prompt = make_prompt(questions)
    response, usage = chat_request(prompt, model, temperature, max_tokens)
    responses.append(response)
    usages.append(usage)
    progress_bar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Generate dummy answers to a list of questions for psychotherapy report test.")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--thread_num", type=int, default=2)
    parser.add_argument("--request_num", type=int, default=8)
    parser.add_argument("--questions_path", type=str, default="questions.json")
    parser.add_argument("--openai_api_key", type=str, default=None)
    args = parser.parse_args()

    if args.openai_api_key:
        openai.api_key = args.openai_api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("Please set OPENAI_API_KEY in environment variable or pass it as argument by `--openai_api_key <your api key>`.")

    questions = json.load(open(args.questions_path, "r"))

    total_cost = 0.0

    threads = []
    global_counter = 0
    with tqdm(total=args.request_num) as progress_bar:

        while global_counter < args.request_num:

            start_time = time.time()

            if args.request_num-global_counter<args.thread_num:
                thread_workers = args.request_num-global_counter
            else:
                thread_workers = args.thread_num

            for _ in range(thread_workers):
                thread = threading.Thread(target=task, args=(questions, progress_bar, args.model, args.temperature, args.max_tokens))
                thread.start()
                threads.append(thread)
    
            for thread in threads:
                thread.join()
        
            print("Saving responses...")

            os.makedirs("answers", exist_ok=True)
            os.makedirs("extracted_failed_answers", exist_ok=True)
            os.makedirs("cleaned_answers", exist_ok=True)
            for response in responses:
                answer = extract_answer(response)
                if answer:
                    with open(f"answers/{len(os.listdir('answers')):03d}.txt", "w") as f:
                        f.write(response)

                    with open(f"cleaned_answers/{len(os.listdir('cleaned_answers')):03d}.txt", "w") as f:
                        f.write(answer)
                else:
                    with open(f"extracted_failed_answers/{len(os.listdir('extracted_failed_answers')):03d}.txt", "w") as f:
                        f.write(response)
    
            # GPT-4 Pricing
            total_prompt_token = sum([usage[0] for usage in usages])
            total_completion_token = sum([usage[1] for usage in usages])
            total_token = total_prompt_token + total_completion_token

            total_cost += (total_prompt_token//1000)*0.03 + (total_completion_token//1000)*0.06
            print(f"\n====================\nTotal token used: {total_token}.\nTotal cost this round: ${(total_prompt_token//1000)*0.03 + (total_completion_token//1000)*0.06}\n====================\n")

            global_counter += len(responses)

            responses.clear()
            usages.clear()

            end_time = time.time()
            if end_time-start_time<120:
                time.sleep(120-(end_time-start_time))

    print("Done all requests.")
    print(f"Total cost: ${total_cost}")

if __name__ == "__main__":
    main()