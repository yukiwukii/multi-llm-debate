from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import re

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

    # I have no idea what's going on. ChatGPT did this.
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
        # print("Some relevant contexts are:")
        # print(context_text)
        # print()
        # print()
        
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

def init(model_list):
    print("Start initializing!")
    for i, model in enumerate(model_list):
        print(f"Model {i+1} turn:")
        answer = model.generate(retrieve=False)
        print(answer)
        print('-' * 50)

        # Store in point form
        model.store_points(answer, metadata={
            "model": model.model_name,
            "persona": model.persona,
            "prompt": model.prompt,
        })
        model.reset_additional_instruction()

def discuss(model_list, n_round=2):
    discussion = ""
    print("Start discussion!")
    for round in range(n_round):
        for i, model in enumerate(model_list):
            print(f"Model {i+1} turn:")
            answer = model.generate(discussion) if discussion else model.generate()

            # Store the entire answer
            model.store(answer, metadata={
                "model": model.model_name,
                "persona": model.persona,
                "prompt": model.prompt,
            })

            discussion += f"Model {i+1}: {answer}\n\n"
            print(answer)
            print('-' * 50)
        print(f"End of round {round + 1}!")
    return discussion

def verdict(model_list, discussion, question):
    print("Start judging round!")
    answers = []
    conclude = 'This is the last round. Give me your final answer to the question, after considering the previous discussions. Provide a short justification of your final answer.'
    
    for i, model in enumerate(model_list):
        print(f"Model {i+1}'s final answer:")
        model.set_system(conclude)
        answer = model.generate(discussion) if discussion else model.generate()
        print(answer)
        print('-' * 50)
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
    print(f'VERDICT BY THE JUDGE IS:\n{verdict_result}')
    return verdict_result


def main():
    question = input('Enter the debate topic: ')
    model_num = input('Enter how many model you want to involve: ')
    model_list = []
    
    for i in range(int(model_num)):
        persona = input(f'Enter persona {i + 1}: ')
        model = RAGModel(
            persona=persona, 
            prompt=question,
            collection_name=f"rag_discussion"
        )
        model_list.append(model)
    
    n_round = input('Enter how many rounds you want to discuss for: ')
    
    init(model_list)
    discussion = discuss(model_list, n_round=int(n_round))
    verdict(model_list, discussion, question)


if __name__ == "__main__":
    main()