from transformers import AutoModel, AutoTokenizer, AutoConfig
from models.roberta import RobertaForV2QuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("/root/reproducibility_code/exps/checkpoint-750")
config = AutoConfig.from_pretrained("/root/reproducibility_code/exps/checkpoint-750")
model = RobertaForV2QuestionAnswering.from_pretrained("/root/reproducibility_code/exps/checkpoint-750", config=config)

tokenizer.push_to_hub("QuoQA-NLP/roberta-reproduction-750step", private=True)
config.push_to_hub("QuoQA-NLP/roberta-reproduction-750step", private=True)
model.push_to_hub("QuoQA-NLP/roberta-reproduction-750step", private=True)