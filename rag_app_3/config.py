from enum import Enum

class ModelProvider(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    MISTRAL = "Mistral AI"
    GROQ = 'Groq'

OPENAI_MODELS = {
    'ChatGPT 4o Mini': 'gpt-4o-mini',
    'ChatGPT 4o': 'gpt-4o',
    'OpenAI o1-preview': 'o1-preview',
    'OpenAI o1-mini': 'o1-mini'
}

ANTHROPIC_MODELS = {
    'Claude 3.5 Sonnet': 'claude-3-5-sonnet-20240620',
    'Claude 3 Opus': 'claude-3-opus-20240229',
    'Claude 3 Haiku': 'claude-3-haiku-20240307'
}

GROQ_MODELS = {
    'Llama 3.1 70B (Preview)': 'llama-3.1-70b-versatile',
    "Llama 3.1 8B (Preview)": "llama-3.1-8b-instant",
    "Llama 3.2 90B (Preview)":"llama-3.2-90b-text-preview"
}

MISTRAL_MODELS = {
    "Mistral NeMo": "open-mistral-nemo-2407",
    "Mistral Large 2": "mistral-large-2407",
    "Mistral Small 24.09": "mistral-small-2409",
    "Codestral": "codestral-2405"
}