def get_model_name(self):
    if self.model == "OLLAMA":
        return "OLLAMA"
    elif self.model == "OPENAI":
        return "OPENAI"
    elif self.model == "CLAUDE":
        return "CLAUDE"
    elif self.model == "GEMINI":
        return "GEMINI"
    elif self.model == "DEEPSEEK":
        return "DEEPSEEK"
    else:
        return "UNKNOWN" 