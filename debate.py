from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import re
import os
import sys
import datetime
import argparse

class RAGModel:
    def __init__(self, model_name="gpt-4o-mini", persona="You are a helpful assistant.",
                 system="You are in a discussion with others. Read previous discussion after the question, and then formulate your answer. DO NOT write answers like 'it depends', you MUST choose an answer.",
                 prompt="What is the capital of Singapore?", additional_instruction="For every argument you make, start with [[POINT]]. Make sure that the argument are standalone sentences, and do not refer to other argument you made.", 
                 retrieval_k=4, collection_name="rag_discussion"):
        self.model_name = model_name
        self.persona = persona
        self.system = system
        self.prompt = prompt
        self.retrieval_k = retrieval_k
        self.collection_name = collection_name
        self.client = OpenAI()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.additional_instruction = additional_instruction
        
        # Init chromadb
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
    
    def get_embedding(self, text):
        if self.embedding_model is None:
            print("Embedding model is not initialized. Figure this out.")
            return [0.0] * 384
            
        return self.embedding_model.encode(text, convert_to_numpy=True).tolist()

    def store(self, text, metadata=None):
        if metadata is None:
            metadata = {}
        
        embedding = self.get_embedding(text)
        doc_id = f"fragment_{self.collection.count()}"
        
        metadata["text"] = text
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text],
            ids=[doc_id]
        )
        return doc_id
    
    def store_points(self, text, metadata=None):
        if metadata is None:
            metadata = {}
        
        stored_ids = []
        points = self.extract_points(text)
        
        if points:
            embeddings = []
            metadatas = []
            documents = []
            ids = []
            
            for i, point in enumerate(points):
                
                point_metadata = metadata.copy()
                point_metadata["text"] = point
                point_metadata["is_point"] = True
                point_metadata["point_index"] = i
                
                point_embedding = self.get_embedding(point)
                point_id = f"point_{self.collection.count()}_{i}"
                
                embeddings.append(point_embedding)
                metadatas.append(point_metadata)
                documents.append(point)
                ids.append(point_id)
                
                stored_ids.append(point_id)
            
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
        
        return stored_ids

    def extract_points(self, text):
        """
        Extract individual points from text marked with [[POINT]] tags at the start of lines.
        Removes the [[POINT]] tag before returning.
        
        :param text: Text containing points
        :return: List of extracted points without [[POINT]] tags
        """
        # Strict pattern to match [[POINT]] at the start of a line, capturing the rest of the line
        # This ensures each point starts with [[POINT]] and takes the entire rest of that line
        pattern = r'^(\[\[POINT\]\]\s*)(.*?)$'
        
        # Use re.MULTILINE to make ^ match the start of each line
        # Remove [[POINT]] and strip any extra whitespace
        points = [re.sub(r'^\[\[POINT\]\]\s*', '', match.group(0)).strip() 
                for match in re.finditer(pattern, text, re.MULTILINE)]
        
        return points
    
    def retrieve(self, query, k=None):
        if k is None:
            k = self.retrieval_k
            
        if self.collection.count() == 0:
            return []
        
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
        )
        
        relevant_contexts = []
        if len(results['ids']) > 0 and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                metadata = self.collection.get(ids=[results['ids'][0][i]])['metadatas'][0]
                if "text" in metadata:
                    relevant_contexts.append({"content": metadata["text"], "metadata": metadata})
        
        return relevant_contexts
    
    def generate(self, discussion='There is no discussion yet. You start first.', retrieve=True):
        relevant_info = self.retrieve(self.prompt) if retrieve else ''

        # Add context text
        context_text = ""
        if relevant_info:
            context_text = "\n\nRelevant information from previous discussions:\n"
            for i, info in enumerate(relevant_info):
                context_text += f"{i+1}. {info['content']}\n"
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.persona + " " + self.system + self.additional_instruction},
                {"role": "user", "content": self.prompt + context_text + "\n\nCurrent discussion:\n" + discussion},
            ],
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
    
    def set_model(self, model_name):
        self.model_name = model_name
    
    def set_prompt(self, prompt):
        self.prompt = prompt
    
    def set_system(self, system):
        self.system = system
    
    def reset_additional_instruction(self):
        self.additional_instruction = ''
    
    def reset_collection(self):
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)


class DebateLogger:    
    def __init__(self, output_dir=None, debate_index=0, topic="debate"):
        self.output_dir = output_dir
        self.debate_index = debate_index
        
        topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_') # ChatGPT did this
        self.filename = f"debate_{debate_index+1}_{topic}.txt"
        
        self.log_file = None
        self.buffer = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.log_file = open(os.path.join(output_dir, self.filename), 'w', encoding='utf-8')
    
    def log(self, message, end='\n'):
        if self.log_file:
            self.log_file.write(message + end)
            self.log_file.flush() # So I can see output immediately
        self.buffer.append(message + end)
    
    def close(self):
        if self.log_file:
            self.log_file.close()


def init(model_list, logger):
    logger.log("Start initializing!")
    for i, model in enumerate(model_list):
        logger.log(f"Model {i+1} turn:")
        answer = model.generate(retrieve=False)
        logger.log(answer)
        logger.log('-' * 50)

        # Store in point form
        model.store_points(answer, metadata={
            "model": model.model_name,
            "persona": model.persona,
            "prompt": model.prompt,
        })
        model.reset_additional_instruction()


def discuss(model_list, n_round=2, logger=None):
    discussion = ""
    logger.log("Start discussion!")
    for round in range(n_round):
        for i, model in enumerate(model_list):
            logger.log(f"Model {i+1} turn:")
            answer = model.generate(discussion) if discussion else model.generate()

            # Store the entire answer
            model.store(answer, metadata={
                "model": model.model_name,
                "persona": model.persona,
                "prompt": model.prompt,
            })

            discussion += f"Model {i+1}: {answer}\n\n"
            logger.log(answer)
            logger.log('-' * 50)
        logger.log(f"End of round {round + 1}!")
    return discussion


def verdict(model_list, discussion, question, logger=None):
    logger.log("Start judging round!")
    answers = []
    conclude = 'This is the last round. Give me your final answer to the question, after considering the previous discussions. Provide a short justification of your final answer.'
    
    for i, model in enumerate(model_list):
        logger.log(f"Model {i+1}'s final answer:")
        model.set_system(conclude)
        answer = model.generate(discussion) if discussion else model.generate()
        logger.log(answer)
        logger.log('-' * 50)
        answers.append(answer)

    judge_system = 'I will provide you with a question and a list of answers. Your task is to determine which is the best. You may choose the one that is most reasonable, most convincing, or most justified. Give a brief justification of your decision.'
    judge_prompt = f'The question is {question}\n'
    
    for i in range(len(answers)):
        judge_prompt += f'Answer {i + 1}: {answers[i]}\n'
    
    judge = RAGModel(
        persona='You are a fair judge.', 
        system=judge_system, 
        prompt=judge_prompt,
        collection_name="rag_judge" # Use a diff collection coz im lazy, its kinda pointless
    )

    judge.reset_additional_instruction()
    verdict_result = judge.generate("Decide which one is the best.")
    logger.log(f'VERDICT BY THE JUDGE IS:\n{verdict_result}')
    return verdict_result


def parse_many(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    debate_configs = content.split('---')
    
    configs = []
    for config_text in debate_configs:
        if config_text.strip():  # Skip empty configurations
            config = parse_one(config_text)
            if config:  # Skip wrong configurations
                configs.append(config)
    
    return configs


def parse_one(config_text):
    config = {}
    lines = config_text.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("DEBATE_TOPIC:"):
            config['question'] = line.split(":", 1)[1].strip()
        elif line.startswith("NUMBER_OF_MODELS:"):
            config['model_num'] = int(line.split(":", 1)[1].strip())
            config['personas'] = []
            for j in range(config['model_num']):
                if i+1+j < len(lines) and lines[i+1+j].strip().startswith("PERSONA_"):
                    persona_line = lines[i+1+j].strip()
                    config['personas'].append(persona_line.split(":", 1)[1].strip())
            i += config['model_num']  # Skip the persona lines
        elif line.startswith("NUMBER_OF_ROUNDS:"):
            config['n_round'] = int(line.split(":", 1)[1].strip())
        i += 1
    
    if 'question' in config and 'model_num' in config and 'personas' in config and 'n_round' in config:
        if len(config['personas']) == config['model_num']:
            return config
    
    return None


def run_debate(config, debate_index=0, output_dir=None):
    logger = DebateLogger(output_dir=output_dir, debate_index=debate_index, topic=config['question'])
    
    logger.log("\n" + "="*80)
    logger.log(f"STARTING DEBATE {debate_index+1}")
    logger.log("="*80 + "\n")
    
    question = config['question']
    model_num = config['model_num']
    model_list = []
    
    logger.log(f"Debate topic: {question}")
    logger.log(f"Number of models: {model_num}")
    
    collection_name = f"rag_discussion_{debate_index}"
    
    for i in range(model_num):
        persona = config['personas'][i]
        logger.log(f"Model {i+1} persona: {persona}")
        
        model = RAGModel(
            persona=persona, 
            prompt=question,
            collection_name=collection_name
        )
        model_list.append(model)
    
    n_round = config['n_round']
    logger.log(f"Number of rounds: {n_round}")
    
    init(model_list, logger)
    discussion = discuss(model_list, n_round=n_round, logger=logger)
    verdict_result = verdict(model_list, discussion, question, logger=logger)
    
    # Clean up
    for model in model_list:
        model.reset_collection()
    logger.close()
    
    return {
        'topic': question,
        'verdict': verdict_result,
        'output_file': os.path.join(output_dir, logger.filename) if output_dir else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Run batch debates from a configuration file.')
    parser.add_argument('config_file', nargs='?', default='configs.txt', 
                        help='Path to the debate configuration file (default: configs.txt)')
    parser.add_argument('--outdir', '-o', nargs='?', default='results', 
                        help='Directory to save debate outputs (default: results')
    
    args = parser.parse_args()
    
    configs = parse_many(args.config_file)
        
    if not configs:
        print("Something is wrong with the config file.")
        return
    
    print(f"Found {len(configs)} debate configs")
    print(f"Output directory: {args.outdir if args.outdir else 'Output directory: ./results.'}")
    
    results = []
    for i, config in enumerate(configs):
        result = run_debate(config, i, args.outdir)
        results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("DEBATE SUMMARY")
    print("="*80)
    
    for i, result in enumerate(results):
        print(f"\nDebate {i+1}: {result['topic']}")
        print(f"Verdict summary: {result['verdict']}...")
        if result['output_file']:
            print(f"Full transcript saved to: {result['output_file']}")
    
    if args.outdir:
        summary_file = os.path.join(args.outdir, f"debate_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DEBATE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Debate {i+1}: {result['topic']}\n")
                f.write(f"Verdict: {result['verdict']}\n")
                f.write(f"Full transcript saved to: {os.path.basename(result['output_file'])}\n\n")
                f.write("-"*80 + "\n\n")
            
    print("\nFinish")


if __name__ == "__main__":
    main()