"""This script use openai gpt model to summarize the qa and make a report for psychotherapy.
"""

import os
import json
import time
import openai
import traceback
import argparse
import threading
from tqdm import tqdm

responses = []
usages = []


def make_prompt(questions: list, answers: list, response_template: dict, return_str: bool = False):
    questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    text = f"""questions:
{questions}

answers:
{answers}

Based on the questions and answers, summarize a detail report for psychotherapist to better and quicker understand the situation about this individual. 
Based on the questions and answers above as a conversation sequence, analyze the background and chat style in great detail using Traditional Chinese.

You always think step-by-step. Be very thorough and explicit.
You make report only from the facts you know.
You always make the report professionally and objectively.

You must respond in Traditional Chinese with valid JSON of form:
```
{json.dumps(response_template, indent=4, ensure_ascii=False)}
```
    """

    if return_str:
        return text

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text}
    ]

    return prompt


def get_all_txt_path(path: str):
    txt_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                txt_path.append(os.path.join(root, file))
    return txt_path


def check_response(response: dict, response_template_path: str = "response_template.json"):
    """Check if the response has the same structure and value type as expected."""
    with open(response_template_path, "r", encoding="utf-8") as template_file:
        template = json.load(template_file)

    # Check if the response dictionary has the same keys as the template
    if set(response.keys()) != set(template.keys()):
        return False

    for key, value in template.items():
        # Check if the response value type matches the template value type
        if not isinstance(response[key], type(value)):
            return False

    return True


def format_response(response: dict) -> str:
    """Format the response dictionary values to a Markdown string."""
    markdown = "# {} 的分析報告\n\n## 個人資料\n{}\n\n## 想要解決的問題\n{}\n\n## 問題的緣由\n{}\n\n## 疾病歷史\n{}\n\n## 個人背景\n{}\n\n## 社交關係\n{}\n\n## 家庭關係\n{}\n\n## 家庭背景\n{}\n\n## 治療目標\n{}\n\n## 遺漏資訊\n{}\n\n---\n\n## 聊天風格\n{}\n\n## 背景描述\n{}\n".format(
        response["report"]["name"],
        response["report"]["personal_information"],
        response["report"]["presenting_issue"],
        response["report"]["issue_history"],
        response["report"]["medical_history"],
        response["report"]["personal_background"],
        response["report"]["social_relationships"],
        response["report"]["family_relationships"],
        response["report"]["family_background"],
        response["report"]["therapy_goals"],
        response["report"]["missing_information"],
        response["chat_style"],
        response["background_description"]
    )
    return markdown


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
             completions['usage']['completion_tokens'])
    return response, usage


def task(filename: str, questions: list, response_template: dict, answers: list, progress_bar: tqdm, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1000):
    prompt = make_prompt(questions, answers, response_template)
    response, usage = chat_request(prompt, model, temperature, max_tokens)
    if not response or not usage:
        return
    responses.append((response, filename))
    usages.append(usage)
    progress_bar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate report for psychotherapy treatment from completed questionnaire.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--thread_num", type=int, default=4)
    parser.add_argument("--questions_path", type=str, default="questions.json")
    parser.add_argument("--answers_path", type=str,
                        default="./cleaned_answers/")
    parser.add_argument("--response_template_path", type=str,
                        default="response_template.json")
    parser.add_argument("--openai_api_key", type=str, default=None)
    args = parser.parse_args()

    if args.openai_api_key:
        openai.api_key = args.openai_api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(
            "Please set OPENAI_API_KEY in environment variable or pass it as argument by `--openai_api_key <your api key>`.")

    questions = json.load(open(args.questions_path, "r"))
    response_template = json.load(
        open(args.response_template_path, "r"))
    answers_path = get_all_txt_path(args.answers_path)

    cost_per_request_estimate = 0.02
    print(
        f"Estimated cost per request: ${cost_per_request_estimate*len(answers_path)}")
    if not input("Continue? (y/n): ").lower().startswith('y'):
        return

    total_cost = 0.0

    threads = []
    global_counter = 0

    os.makedirs("reports", exist_ok=True)
    os.makedirs("extracted_failed_reports", exist_ok=True)
    os.makedirs("cleaned_reports", exist_ok=True)

    # Check if the report is already generated in cleaned_reports or extracted_failed_reports, if exist then remove the path from answers_path list.
    for path in answers_path.copy():
        filename = path.split("/")[-1]
        if os.path.exists(f"cleaned_reports/{filename}") or os.path.exists(f"extracted_failed_reports/{filename}"):
            answers_path.remove(path)

    total_cost_list = []
    general_timer = time.time()
    with tqdm(total=len(answers_path)) as progress_bar:

        while global_counter < len(answers_path):

            start_time = time.time()

            if len(answers_path)-global_counter < args.thread_num:
                thread_workers = len(answers_path)-global_counter
            else:
                thread_workers = args.thread_num

            for i in range(thread_workers):
                answer = open(answers_path[global_counter+i], "r").readlines()
                filename = answers_path[global_counter+i].split("/")[-1]

                thread = threading.Thread(target=task, args=(
                    filename, questions, response_template, answer, progress_bar, args.model, args.temperature, args.max_tokens))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            print("Saving responses...")
            for response, filename in responses:
                # Check if reponse is a valid JSON string.
                try:
                    response_json = json.loads(response)
                except:
                    with open(f"extracted_failed_reports/{filename}", "w") as f:
                        f.write(response)
                    continue

                # Check if response has the same structure and value type as expected.
                if not check_response(response_json):
                    with open(f"extracted_failed_reports/{filename}", "w") as f:
                        f.write(response)
                    continue

                markdown = format_response(response_json)

                answer_filename = filename.replace(".txt", ".json")
                with open(f"reports/{answer_filename}", "w", encoding='utf-8') as f:
                    f.write(json.dumps(response_json, indent=4,
                            ensure_ascii=False))

                md_filename = filename.replace(".txt", ".md")
                with open(f"cleaned_reports/{md_filename}", "w", encoding="utf-8") as f:
                    f.write(markdown)

            # gpt-3.5-turbo Pricing
            total_prompt_token = sum([usage[0] for usage in usages])
            total_completion_token = sum([usage[1] for usage in usages])
            total_token = total_prompt_token + total_completion_token

            total_cost += (total_prompt_token//1000)*0.002 + \
                (total_completion_token//1000)*0.002
            total_cost_list.append(
                (total_prompt_token//1000)*0.002 + (total_completion_token//1000)*0.002)
            print(
                f"\n====================\nTotal token used: {total_token}.\nTotal cost this round: ${(total_prompt_token//1000)*0.002 + (total_completion_token//1000)*0.002}")

            global_counter += len(responses)

            responses.clear()
            usages.clear()

            end_time = time.time()
            print(
                f"Time spent: {end_time-start_time} seconds.\n====================\n")

    print(
        f"====================\nDone all.\nTotal cost: ${total_cost}\nAverage cost per report: ${sum(total_cost_list)/len(total_cost_list)}\nTotal time spent: {time.time()-general_timer} seconds.\n====================\n")


if __name__ == "__main__":
    main()
