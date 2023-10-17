import re
import time
from copy import copy
from os import environ
from rich import print

import tiktoken
import openai

from .base_translator import Base

PROMPT_ENV_MAP = {
    "user": "BBM_CHATGPTAPI_USER_MSG_TEMPLATE",
    "system": "BBM_CHATGPTAPI_SYS_MSG",
}


class ChatGPTAPI(Base):
    DEFAULT_PROMPT = "Please help me to translate,`{text}` to {language}, please return only translated content not include the origin text"

    def __init__(
        self,
        key,
        language,
        api_base=None,
        prompt_template=None,
        prompt_sys_msg=None,
        temperature=1.0,
        **kwargs,
    ) -> None:
        super().__init__(key, language)
        self.key_len = len(key.split(","))

        if api_base:
            openai.api_base = api_base
        self.prompt_template = (
            prompt_template
            or environ.get(PROMPT_ENV_MAP["user"])
            or self.DEFAULT_PROMPT
        )
        self.prompt_sys_msg = (
            prompt_sys_msg
            or environ.get(
                "OPENAI_API_SYS_MSG",
            )  # XXX: for backward compatibility, deprecate soon
            or environ.get(PROMPT_ENV_MAP["system"])
            or ""
        )
        self.system_content = environ.get("OPENAI_API_SYS_MSG") or ""
        self.deployment_id = None
        self.temperature = temperature
        self.completion_spent = 0
        self.prompt_spent = 0

    def rotate_key(self):
        openai.api_key = next(self.keys)

    def count_tokens(self,string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def create_chat_completion_2pass(self, text):
        system_message = "You are a professional translator. You always translate accurately, fluently and reliably."
        # user_message = f"Translate to {self.language}, return only translated content, don't include original text. Text to be translated:\n{text}"
        translation_prompt=f'''
Translation Guideline:
- Retain specific terms/names, put it after the translation in brackets, for example: "乔（Joe）".
- Divide the translation into two parts and print each result:
1. Translate directly based on the content, without omitting any information.
2. Based on the first direct translation, rephrase it to make the content more easily understood and conform to {self.language} expression habits, while adhering to the original meaning.
Without any comment, return the result in the following python dict format:
[{{"direct_translation": "direct translation here",
"better_translation": "better translation here",}}]
Reply OK to this message and I'll send you text to be translated to {self.language} afterwards.'''

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": translation_prompt},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": f'{{"text": "{text}","target_language": "{self.language}",}}'},
        ]

        if self.deployment_id:
            return openai.ChatCompletion.create(
                engine=self.deployment_id,
                messages=messages,
                temperature=self.temperature,
            )

        text_length=self.count_tokens(system_message+translation_prompt+text+"OK")
        if text_length<=1200:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=self.temperature,
            )
        else:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                temperature=self.temperature,
            )

    def create_chat_completion(self, text):
        content = self.prompt_template.format(
            text=text, language=self.language, crlf="\n"
        )
        sys_content = self.system_content or self.prompt_sys_msg.format(crlf="\n")
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": content},
        ]

        if self.deployment_id:
            return openai.ChatCompletion.create(
                engine=self.deployment_id,
                messages=messages,
                temperature=self.temperature,
            )

        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=self.temperature,
        )
    
    def get_translation(self, text):
        self.rotate_key()

        completion = {}
        try:
            completion = self.create_chat_completion_2pass(text)
        except Exception:
            if (
                "choices" not in completion
                or not isinstance(completion["choices"], list)
                or len(completion["choices"]) == 0
            ):
                raise
            if completion["choices"][-1]["finish_reason"] != "length":
                raise
        
        # Calculate fare
        if '16k' in completion['model']:
            self.prompt_spent+=completion['usage']['prompt_tokens']*3
            self.completion_spent+=completion['usage']['completion_tokens']*4
        else:
            self.prompt_spent+=completion['usage']['prompt_tokens']*1.5
            self.completion_spent+=completion['usage']['completion_tokens']*2
        print(f"Prompt spent: ${self.prompt_spent/1e6}\nCompletion spent: ${self.completion_spent/1e6}\nTotal spent: ${(self.prompt_spent+self.completion_spent)/1e6}")

        # work well or exception finish by length limit
        choice = completion["choices"][-1]

        response_text = choice.get("message").get("content", "").encode("utf8").decode()
        pattern = r'"better_translation":[\s\n]*[“"”]([\s\S]*?)[“"”]?,?[\s\n]*}'
        match = re.search(pattern, response_text)
        
        if match:
            t_text = match.group(1)
        else:
            t_text = response_text

        if choice["finish_reason"] == "length":
            with open("log/long_text.txt", "a") as f:
                print(
                    f"""==================================================
The total token is too long and cannot be completely translated\n
{text}
""",
                    file=f,
                )
        
        return t_text.replace('\\n', '\n')

    def translate(self, text, needprint=True):
        start_time = time.time()
        # todo: Determine whether to print according to the cli option
        if needprint:
            print(re.sub("\n{3,}", "\n\n", text))

        attempt_count = 0
        max_attempts = 3
        t_text = ""

        while attempt_count < max_attempts:
            try:
                t_text = self.get_translation(text)
                if '"direct_trans":' in t_text and attempt_count==0: 
                    print(f"Response illegal, retrying...\nResponse={t_text}")
                    attempt_count += 1
                    continue # if failed to capture 2pass result for some reason retry
                elif '_trans":' in t_text and attempt_count>0:
                    t_text = t_text.split('_trans":')[-1]
                break
            except Exception as e:
                # todo: better sleep time? why sleep alawys about key_len
                # 1. openai server error or own network interruption, sleep for a fixed time
                # 2. an apikey has no money or reach limit, don`t sleep, just replace it with another apikey
                # 3. all apikey reach limit, then use current sleep
                sleep_time = int(60 / self.key_len)
                print(e, f"will sleep {sleep_time} seconds")
                time.sleep(sleep_time)
                attempt_count += 1
                if attempt_count == max_attempts:
                    print(f"Get {attempt_count} consecutive exceptions")
                    raise

        # todo: Determine whether to print according to the cli option
        if needprint:
            print("[bold green]" + re.sub("\n{3,}", "\n\n", t_text) + "[/bold green]")

        time.time() - start_time
        # print(f"translation time: {elapsed_time:.1f}s")

        return t_text

    def translate_and_split_lines(self, text):
        result_str = self.translate(text, False)
        lines = result_str.splitlines()
        lines = [line.strip() for line in lines if line.strip() != ""]
        return lines

    def get_best_result_list(
        self,
        plist_len,
        new_str,
        sleep_dur,
        result_list,
        max_retries=15,
    ):
        if len(result_list) == plist_len:
            return result_list, 0

        best_result_list = result_list
        retry_count = 0

        while retry_count < max_retries and len(result_list) != plist_len:
            print(
                f"bug: {plist_len} -> {len(result_list)} : Number of paragraphs before and after translation",
            )
            print(f"sleep for {sleep_dur}s and retry {retry_count+1} ...")
            time.sleep(sleep_dur)
            retry_count += 1
            result_list = self.translate_and_split_lines(new_str)
            if (
                len(result_list) == plist_len
                or len(best_result_list) < len(result_list) <= plist_len
                or (
                    len(result_list) < len(best_result_list)
                    and len(best_result_list) > plist_len
                )
            ):
                best_result_list = result_list

        return best_result_list, retry_count

    def log_retry(self, state, retry_count, elapsed_time, log_path="log/buglog.txt"):
        if retry_count == 0:
            return
        print(f"retry {state}")
        with open(log_path, "a", encoding="utf-8") as f:
            print(
                f"retry {state}, count = {retry_count}, time = {elapsed_time:.1f}s",
                file=f,
            )

    def log_translation_mismatch(
        self,
        plist_len,
        result_list,
        new_str,
        sep,
        log_path="log/buglog.txt",
    ):
        if len(result_list) == plist_len:
            return
        newlist = new_str.split(sep)
        with open(log_path, "a", encoding="utf-8") as f:
            print(f"problem size: {plist_len - len(result_list)}", file=f)
            for i in range(len(newlist)):
                print(newlist[i], file=f)
                print(file=f)
                if i < len(result_list):
                    print("............................................", file=f)
                    print(result_list[i], file=f)
                    print(file=f)
                print("=============================", file=f)

        print(
            f"bug: {plist_len} paragraphs of text translated into {len(result_list)} paragraphs",
        )
        print("continue")

    def join_lines(self, text):
        lines = text.splitlines()
        new_lines = []
        temp_line = []

        # join
        for line in lines:
            if line.strip():
                temp_line.append(line.strip())
            else:
                if temp_line:
                    new_lines.append(" ".join(temp_line))
                    temp_line = []
                new_lines.append(line)

        if temp_line:
            new_lines.append(" ".join(temp_line))

        text = "\n".join(new_lines)

        # del ^M
        text = text.replace("^M", "\r")
        lines = text.splitlines()
        filtered_lines = [line for line in lines if line.strip() != "\r"]
        new_text = "\n".join(filtered_lines)

        return new_text

    def translate_list(self, plist):
        sep = "\n\n\n\n\n"
        # new_str = sep.join([item.text for item in plist])

        new_str = ""
        i = 1
        for p in plist:
            temp_p = copy(p)
            for sup in temp_p.find_all("sup"):
                sup.extract()
            new_str += f"({i}) {temp_p.get_text().strip()}{sep}"
            i = i + 1

        if new_str.endswith(sep):
            new_str = new_str[: -len(sep)]

        new_str = self.join_lines(new_str)

        plist_len = len(plist)

        print(f"plist len = {len(plist)}")

        result_list = self.translate_and_split_lines(new_str)

        start_time = time.time()

        result_list, retry_count = self.get_best_result_list(
            plist_len,
            new_str,
            6,
            result_list,
        )

        end_time = time.time()

        state = "fail" if len(result_list) != plist_len else "success"
        log_path = "log/buglog.txt"

        self.log_retry(state, retry_count, end_time - start_time, log_path)
        self.log_translation_mismatch(plist_len, result_list, new_str, sep, log_path)

        # del (num), num. sometime (num) will translated to num.
        result_list = [re.sub(r"^(\(\d+\)|\d+\.|(\d+))\s*", "", s) for s in result_list]
        return result_list

    def set_deployment_id(self, deployment_id):
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        self.deployment_id = deployment_id
