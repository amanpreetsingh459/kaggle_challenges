# Google - Unlock Global Communication with Gemma

[kaggle challenge link](https://www.kaggle.com/competitions/gemma-language-tuning)

This model is a fine-tuned version of [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
gemma_tokenizer = AutoTokenizer.from_pretrained("amanpreetsingh459/gemma-2-2b-punjabi-finetuned-4")
EOS_TOKEN = gemma_tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "amanpreetsingh459/gemma-2-2b-punjabi-finetuned-4",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = gemma_tokenizer(
[
    alpaca_prompt.format(
        "ਮੇਨੂ ਏਕ ਕਵਿਤਾ ਲਿੱਖ ਕੇ ਦੇਯੋ ਜੀ", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 250)
decoded_outputs = gemma_tokenizer.batch_decode(outputs)
print(decoded_outputs[0])

```
## Download the model
- [Huggingface link](https://huggingface.co/amanpreetsingh459/gemma-2-2b-punjabi-finetuned-4)
- [kaggle link](https://www.kaggle.com/models/amankaggle57/gemma-2-2b-punjabi-finetuned)

## Files(in the repo) description
- **[training_gemma_2_2b_punjabi_finetuned.ipynb](https://github.com/amanpreetsingh459/kaggle_challenges/blob/master/google-gemma-2-language-fine-tuning/training_gemma_2_2b_punjabi_finetuned.ipynb)**: This notebook contains the entire code to fine-tune the model and make inference
- **[inference_gemma_2_2b_punjabi_finetuned.ipynb](https://github.com/amanpreetsingh459/kaggle_challenges/blob/master/google-gemma-2-language-fine-tuning/inference_gemma_2_2b_punjabi_finetuned.ipynb)**: This notebook contains the code to only make inference
- **[evaluate.ipynb](https://github.com/amanpreetsingh459/kaggle_challenges/blob/master/google-gemma-2-language-fine-tuning/evaluate.ipynb)**: This notebook contains the evaluation methods to evaluate the model on varous metrics such as BLEU, ROUGE etc.
- **[loss_table.csv](https://github.com/amanpreetsingh459/kaggle_challenges/blob/master/google-gemma-2-language-fine-tuning/loss_table.csv)**: This file contains the loss table of the model while fine-tuning for 1 epoch(44000 steps)
- **[test_data_predictions.csv](https://github.com/amanpreetsingh459/kaggle_challenges/blob/master/google-gemma-2-language-fine-tuning/test_data_predictions.csv)**: This file has the test data and it's generated predictions on which the model has been evaluated.

## Citations
    
```bibtex
@article{gemma_2024,
    title={Gemma},
    url={https://www.kaggle.com/m/3301},
    DOI={10.34740/KAGGLE/M/3301},
    publisher={Kaggle},
    author={Gemma Team},
    year={2024}
}

@misc{gemma-language-tuning,
    author = {Glenn Cameron and Lauren Usui and Paul Mooney and Addison Howard},
    title = {Google - Unlock Global Communication with Gemma},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/gemma-language-tuning}},
    note = {Kaggle}
}

@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}

@misc{PunjabiAlpaca,
  author = {Sambit Sekhar and Shantipriya Parida},
  title = {Punjabi Instruction Set Based on Alpaca},
  year = {2023},
  publisher = {Hugging Face},
  journal = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/OdiaGenAI}},
}

```
