import torch._dynamo.config
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_hub import login
import argparse
import torch
from dataset import CorrectionJSONDataset
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# for utilizing GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#### argument #### 
parser = argparse.ArgumentParser()
# required = True
parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=True, default="google/gemma-2b")
parser.add_argument('--dataset_path', help='path of correction dataset', required=True, type=str)
parser.add_argument('--per_device_train_batch_size', help='total batch size / # of gradient accumulation steps', type=int, required=True, default=4)
parser.add_argument('--gradient_accumulation_steps', help='# of gradient accumulation steps', type=int, required=True, default=8)
# required = False
parser.add_argument('--save_path', help='path where the aligner model ckpt to be saved', type=str, required=False, default='./models')
parser.add_argument('--logging_dir', help='path where the logging of aligner model to be saved', type=str, required=False, default="./runs")
parser.add_argument('--lr', help='learning rate', type=float, required=False, default=2e-5)
parser.add_argument('--lr_scheduler_type', help='learning rate scheduler', type=str, required=False, default="cosine")
parser.add_argument('--epoch', help='training epoch', type=int, required=False, default=3)
parser.add_argument('--lr_warmup_ratio', help='warmup step ratio, which is # of steps ("total steps * ratio")', type=float, required=False, default=0.03)
parser.add_argument('--weight_decay', help='weight decay', type=float, required=False, default=0.0)
parser.add_argument('--huggingface_api_key', help='huggingface api key for gemma, llama ...', type=str, required=False)
args = parser.parse_args()



dataset_path = args.dataset_path
model_ckpt = args.model_ckpt
save_path = args.save_path
logging_dir = args.logging_dir
lr = args.lr
lr_scheduler_type = args.lr_scheduler_type
epoch = args.epoch
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
lr_warmup_ratio = args.lr_warmup_ratio
weight_decay = args.weight_decay

### main code ###
api_key = args.huggingface_api_key
login(token=api_key)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

tokenizer.padding_side = "right" # standard
base_model = AutoModelForCausalLM.from_pretrained(model_ckpt)
print(f"### loaded PLM ###")

dataset = CorrectionJSONDataset(dataset_path, tokenizer)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
print("### loaded dataset ###")

training_args = TrainingArguments(
    output_dir=save_path,
    logging_strategy='steps',
    logging_steps=5,
    torch_compile=True,
    save_strategy="no",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=lr_warmup_ratio,
    weight_decay=weight_decay,
    seed=42,
    report_to='tensorboard',
    logging_dir=logging_dir,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=base_model.to(device),
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print('### start fine-tuning ###')
base_model.config.use_cache=False
trainer.train()
trainer.save_state()
trainer.save_model()
print('### ended fine-tuning ###')