"""This script use openai gpt model to summarize the qa and make a report for psychotherapy.
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

def make_prompt(questions: list, answers: list, return_str: bool = False):
    questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    text = f"""The following is a conversation with an AI assistant.

The assistant summarize the report for psychotherapist better and quicker understand the situation about this individual in psychotherapy treatment.

The assistant must return the following structure with Markdown format:

RESPONSE
```markdown
## 個人資料
- your report.

## 想要解決的問題
- your report.

## 問題的緣由
- your report.

## 疾病歷史
- your report.

## 個人和社交背景
- your report.

## 家庭背景
- your report.

## 治療目標
- your report.
```

questions:
{questions}

answers:
{answers}

program:

- Based on the reponse markdown template, what facts do you need to look for in questions and answers to make the report for psychotherapist better and quicker understand the situation about this individual?
- does the questions and answers contain all the facts needed to make the report and contain all information that's helpful during psychotherapy treatment?
- think about how you make the report in detail given what you know step-by-step.
- Make the report into the given Markdown format using Traditional Chinese. Return the tag RESPONSE just before your report.

State each step of the program and show your work for performing that step.

1: Based on the reponse markdown template, what facts do you need to look for in questions and answers to make the report for psychotherapist better and quicker understand the situation about this individual?

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
    usage = completions['usage']['total_tokens']
    return response, usage

def task(filename: str, questions: list, answers: list, progress_bar: tqdm, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
    prompt = make_prompt(questions, answers)
    response, usage = chat_request(prompt, model, temperature, max_tokens)
    if not response or not usage:
        return
    responses.append((response, filename))
    usages.append(usage)
    progress_bar.update(1)

def get_all_txt_path(path: str):
    txt_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                txt_path.append(os.path.join(root, file))
    return txt_path

def main():
    parser = argparse.ArgumentParser(description="Generate report for psychotherapy treatment from completed questionnaire.")
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--thread_num", type=int, default=2)
    parser.add_argument("--questions_path", type=str, default="questions.json")
    parser.add_argument("--answers_path", type=str, default="./cleaned_answers/")
    parser.add_argument("--openai_api_key", type=str, default=None)
    args = parser.parse_args()

    if args.openai_api_key:
        openai.api_key = args.openai_api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("Please set OPENAI_API_KEY in environment variable or pass it as argument by `--openai_api_key <your api key>`.")

    questions = json.load(open(args.questions_path, "r"))
    answers_path = get_all_txt_path(args.answers_path)

    total_cost = 0.0

    threads = []
    global_counter = 0
    with tqdm(total=len(answers_path)) as progress_bar:

        while global_counter < len(answers_path):

            start_time = time.time()

            if len(answers_path)-global_counter<args.thread_num:
                thread_workers = len(answers_path)-global_counter
            else:
                thread_workers = args.thread_num

            for i in range(thread_workers):
                answer = open(answers_path[global_counter+i], "r").readlines()
                filename = answers_path[global_counter+i].split("/")[-1]

                thread = threading.Thread(target=task, args=(filename, questions, answer, progress_bar, args.model, args.temperature, args.max_tokens))
                thread.start()
                threads.append(thread)
    
            for thread in threads:
                thread.join()
        
        
            print("Saving responses...")

            os.makedirs("reports", exist_ok=True)
            os.makedirs("extracted_failed_reports", exist_ok=True)
            os.makedirs("cleaned_reports", exist_ok=True)
            for response, filename in responses:
                ans = extract_answer(response)
                if ans:
                    with open(f"reports/{filename}", "w") as f:
                        f.write(response)

                    with open(f"cleaned_reports/{filename}", "w") as f:
                        f.write(ans)
                else:
                    with open(f"extracted_failed_reports/{filename}", "w") as f:
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

    print("Done all processes.")
    print(f"Total cost: ${total_cost}")

if __name__ == "__main__":
    main()