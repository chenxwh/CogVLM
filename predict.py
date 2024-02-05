# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        cache_dir = "model_cache"
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
            cache_dir=cache_dir,
        )
        self.torch_type = torch.bfloat16
        self.device = "cuda"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                "THUDM/cogagent-chat-hf",
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            .to(self.device)
            .eval()
        )

    def predict(
        self,
        query: str = Input(description="Input query.", default="Describe this image."),
        image: Path = Input(description="Input image."),
        temperature: float = Input(
            default=0.9,
            description="Adjusts randomness of textual outputs, greater than 1 is random and 0 is deterministic.",
        ),
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image)).convert("RGB")

        model = self.model
        input_by_model = model.build_conversation_input_ids(
            self.tokenizer, query=query, history=[], images=[image]
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.device),
            "images": [
                [input_by_model["images"][0].to(self.device).to(self.torch_type)]
            ],
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(self.device).to(self.torch_type)]
            ]

        gen_kwargs = {
            "max_length": 2048,
            "do_sample": True,
            "temperature": temperature,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(outputs[0]).strip()
        if response.endswith("</s>"):
            response = response[: -len("</s>")]
        return response
