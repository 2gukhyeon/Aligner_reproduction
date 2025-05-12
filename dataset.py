import jsonlines
from torch.utils.data import Dataset
from preprocess import preprocess
from transformers import AutoTokenizer

# input format
correction_instruction = '{question} | {answer}'


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'


class CorrectionJSONDataset(Dataset):
    def __init__(self, path, tokenizer): 
        self.path = path
        self.data = []
        self.tokenizer = tokenizer
        
        with jsonlines.open(self.path) as f:
            for line in f.iter():
                self.data.append(line)
      
        self.prepare_sft_dataset()
    
    def tokenizing(self, lines):
        encoding = self.tokenizer(
            lines,
            return_tensors = "pt",
            add_special_tokens=True,
            padding=True,
            truncation="longest_first",
            max_length=2048
        )
        
        return encoding
    
    def prepare_sft_dataset(self, verbose=False):
        prompts = [preprocess(correction_instruction.format(question=item["question"], answer=item["answer"]), self.tokenizer.eos_token) for item in self.data]
        answers = [str(item["correction"]) for item in self.data]
        
        texts = [str(prompt) + str(answer) + self.tokenizer.eos_token for prompt, answer in zip(prompts, answers)]
        
        self.encoding = self.tokenizing(texts)
        input_ids = self.encoding["input_ids"]
        self.labels = input_ids.clone()
        print(self.tokenizing(prompts[0])["input_ids"][0])

        for idx in range(len(self.labels)):
            self.labels[idx,:len(self.tokenizing(prompts[idx])['input_ids'][0])] = IGNORE_INDEX

        if verbose:
            print(f'sample example: {texts[0]}')
            
        return None
        
    def __getitem__(self, index):
        data = {key: val[index] for key, val in self.encoding.items()}
        # data['labels'] = self.labels['input_ids'][index]
        data['labels'] = self.labels[index]
        return data 

    def __len__(self) -> int:
        return len(self.labels)



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    # tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.eos_token_id)
    tokenizer.padding_side = "right" # standard
    data = CorrectionJSONDataset("./dataet/aligner_train.jsonl", tokenizer)
    print(data[0])