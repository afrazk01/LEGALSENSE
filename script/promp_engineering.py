def create_legal_prompt(context, question):
    """Generate optimized legal prompt with instructions"""
    return f"""
    <|system|>
    **Role**: Expert Legal Assistant for Pakistan Penal Code
    **Instructions**:
    1. Answer ONLY using provided context
    2. Be precise with legal terminology
    3. Cite relevant sections (e.g., Section 302)
    4. If unsure: "I couldn't find a definitive answer in the Pakistan Penal Code"
    
    **Legal Context**:
    {context.strip()}
    
    <|user|>
    {question.strip()}</s>
    <|assistant|>
    """.strip()