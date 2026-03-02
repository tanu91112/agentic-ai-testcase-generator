import json
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# =========================
# Load Local LLM (Flan-T5 Small)
# =========================

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()
torch.set_grad_enabled(False)


# =========================
# LLM Generation Function
# =========================

def generate_with_llm(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    """
    Local LLM generation using Flan-T5.
    No API key required.
    Runs fully offline.
    """

    prompt = f"""
    {system_prompt}

    {user_prompt}
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_length=400,
        temperature=0.3,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# Prompt Builders
# =========================

def build_test_case_prompt(context_chunks: List[str], user_query: str) -> str:
    joined = '\n\n---\n\n'.join(context_chunks)

    prompt = f"""You are a QA engineer.
Use ONLY the following context.
Do not hallucinate.

Context:
{joined}

User Request:
{user_query}

Return structured test cases in JSON array format.
Each test case must contain:
- Test_ID
- Feature
- Test_Scenario
- Steps (array)
- Expected_Result
- Grounded_In (source reference)
"""

    return prompt


def build_script_prompt(html_content: str, selected_test_case: dict, context_chunks: List[str]) -> str:
    joined = '\n\n---\n\n'.join(context_chunks)

    prompt = f"""You are a Selenium Python expert.
Use ONLY the provided HTML and context.

HTML:
{html_content}

Context:
{joined}

Test Case:
{json.dumps(selected_test_case, indent=2)}

Requirements:
- Use webdriver.Chrome()
- Use explicit waits (WebDriverWait)
- Use valid selectors from the HTML (id, name, CSS)
- Script must run as: python script.py

Return ONLY the Python script.
No explanations.
"""

    return prompt