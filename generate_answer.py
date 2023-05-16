"""This script use openai gpt model to generate multiple dummy answers to a list of questions.
"""

import os
import re
import json
import time
import openai
import random
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
    patterns = [r"RESPONSE\n(.*?)```markdown(.*?)```",
                r"```markdown(.*?)```", r"```(.*?)```"]

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


def make_prompt(questions: list, character: str, answers_template: dict = "answers_template.txt", return_str: bool = False):
    questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    text = f"""questions:
{questions}

character:
{character}

You always think step-by-step. Be very creative.
Make up a detail mental trauma that cause the character to seek psychotherapy.
Make up a detail personal information include personal history, personal background, family history, family background, social relationships.
Make up a detail chat style of this character that is unique for this character based on its background.
Act as this character and reply each questions with its chat style you defined strictly in Traditional Chinese. 
Store all replies into an array, do not mark bullet or number for replies, you should have 14 replies.

You must respond in Traditional Chinese with valid JSON of form:
```
{json.dumps(answers_template, indent=4, ensure_ascii=False)}
```"""

    if return_str:
        return text

    prompt = [
        {"role": "system", "content": "You are a helpful assistant, you can deal with both Traditional Chinese and English well."},
        {"role": "user", "content": text}
    ]

    return prompt


def check_response(response: dict, answers_template: dict):
    """Check if response has the same structure and value type as expected."""
    if not isinstance(response, dict):
        return False

    if not response.keys() == answers_template.keys():
        return False

    # TODO: Need deeper check.

    if len(response["reply"]) != 14:
        print(f"Expect 14 replies, but got {len(response['reply'])}.")
        return False

    return True


def format_response(response: list) -> str:
    """Format the response list values to markdown number list format"""
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(response)])


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
    usage = (completions['usage']['prompt_tokens'],
             completions['usage']['completion_tokens'],
             completions['usage']['total_tokens'])
    return response, usage


def task(questions: list, character: str, answers_template: dict, progress_bar: tqdm, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
    prompt = make_prompt(questions, character, answers_template)
    response, usage = chat_request(prompt, model, temperature, max_tokens)
    responses.append(response)
    usages.append(usage)
    progress_bar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dummy answers to a list of questions for psychotherapy report test.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--thread_num", type=int, default=8)
    parser.add_argument("--request_num", type=int, default=10)
    parser.add_argument("--questions_path", type=str, default="questions.json")
    parser.add_argument("--characters_path", type=str,
                        default="characters.json")
    parser.add_argument("--answers_template_path", type=str,
                        default="answers_template.json")
    parser.add_argument("--openai_api_key", type=str, default=None)
    args = parser.parse_args()

    if args.openai_api_key:
        openai.api_key = args.openai_api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(
            "Please set OPENAI_API_KEY in environment variable or pass it as argument by `--openai_api_key <your api key>`.")

    cost_per_request_estimate = 0.02
    print(
        f"Estimated cost per request ({args.model}): ${cost_per_request_estimate*args.request_num}")
    if not input("Continue? (y/n): ").lower().startswith('y'):
        return

    questions = json.load(open(args.questions_path, "r"))
    answers_template = json.load(open(args.answers_template_path, "r"))
    characters = json.load(open(args.characters_path, "r"))
    random.shuffle(characters)

    if args.request_num > len(characters):
        raise ValueError(
            f"Request number ({args.request_num}) is larger than the number of characters ({len(characters)}).")

    total_cost = 0.0

    threads = []
    global_counter = 0
    total_cost_list = []
    general_timer = time.time()
    with tqdm(total=args.request_num) as progress_bar:

        while global_counter < args.request_num:

            failed_counter = 0

            start_time = time.time()

            if args.request_num-global_counter < args.thread_num:
                thread_workers = args.request_num-global_counter
            else:
                thread_workers = args.thread_num

            for i in range(thread_workers):
                character = characters[global_counter+i]

                thread = threading.Thread(target=task, args=(
                    questions, character, answers_template, progress_bar, args.model, args.temperature, args.max_tokens))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            print("Saving responses...")

            os.makedirs("answers", exist_ok=True)
            os.makedirs("extracted_failed_answers", exist_ok=True)
            os.makedirs("cleaned_answers", exist_ok=True)
            for response, usage in zip(responses, usages):
                # Check if reponse is a valid JSON string.
                try:
                    response_json = json.loads(response)
                except:
                    failed_counter += 1
                    with open(f"extracted_failed_answers/{len(os.listdir('extracted_failed_answers')):03d}.txt", "w") as f:
                        f.write(response)
                    total_token = sum([usage[2] for usage in usages])
                    total_cost += (total_token//1000)*0.002
                    total_cost_list.append((total_token//1000)*0.002)
                    continue

                # Check if response has the same structure and value type as expected.
                if not check_response(response_json, answers_template):
                    failed_counter += 1
                    with open(f"extracted_failed_answers/{len(os.listdir('extracted_failed_answers')):03d}.txt", "w") as f:
                        f.write(response)
                    total_token = sum([usage[2] for usage in usages])
                    total_cost += (total_token//1000)*0.002
                    total_cost_list.append((total_token//1000)*0.002)
                    continue

                answers = response_json["reply"]
                answers_write_ready = format_response(answers)

                with open(f"answers/{len(os.listdir('answers')):03d}.txt", "w", encoding='utf-8') as f:
                    f.write(json.dumps(response_json, indent=4,
                            ensure_ascii=False))

                with open(f"cleaned_answers/{len(os.listdir('cleaned_answers')):03d}.txt", "w") as f:
                    f.write(answers_write_ready)

            # gpt-3.5-turbo Pricing
            total_token = sum([usage[2] for usage in usages])

            total_cost += (total_token//1000)*0.002
            total_cost_list.append((total_token//1000)*0.002)
            print(
                f"\n====================\nTotal token used: {total_token}.\nTotal cost this round: ${(total_token//1000)*0.002}")

            global_counter += len(responses) - failed_counter

            responses.clear()
            usages.clear()

            end_time = time.time()
            print(
                f"Time spent: {end_time-start_time} seconds.\n====================\n")

    print(
        f"====================\nDone all.\nTotal cost: ${total_cost}\nAverage cost per answer: ${sum(total_cost_list)/len(total_cost_list)}\nTotal time spent: {time.time()-general_timer} seconds.\n====================\n")


if __name__ == "__main__":
    main()
