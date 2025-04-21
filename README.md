# multi LLM debate
## 🌏 Overview
Are you like me? Do you also have an unquenchable desire to watch people fight? LLMs are too helpful these days, we need to make them more toxic. That's why I built `multi LLM debate`. ~~Currently, it is running strictly on CLI, but maybe one day I will make it better. Maybe not.~~ You can now input through `.txt` file and it will output to a specified `--outdir`.

Basically, our debate consists of 3 stages:
1. **Initialization**. Each LLM are given the opportunity to make their point without being affected by other LLMs. (One Round).
2. **Debate**. At each LLM turn, it will use RAG to retrieve relevant previous discussion points, then respond to it. (You can specify how many rounds this will go for).
3. **Verdict**. Each LLM can make their closing comments and a judge LLM will determine which one is the best answer. (One Round).

## 🚀 Quick Launch
1. Clone repo and install requirements.
```bash
# Clone the repository
git clone https://github.com/yukiwukii/multi-llm-debate.git

# Move into the repository directory
cd multi-llm-debate

# Optional: Create a virtual environment
# conda create -n llmdebate python=3.11
# conda activate llmdebate

# Install the dependencies
pip install -r requirements.txt
```
2. Create a `.env` file consisting of your OpenAI API Key.
```bash
OPENAI_API_KEY = 'your_api_key_here'
```
3. Input your configurations (see `configs.txt` for an example).
4. Run the only script and have fun!
```bash
python debate.py configs.txt --outdir output

# Results will be stored at ./output/
```