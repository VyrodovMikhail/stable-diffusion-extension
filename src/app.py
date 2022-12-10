import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed
from random import randint
import re
import streamlit as st

input_text = "The pink city in the sky under beautiful sun"


def generate_prompts(starting_text):
    model_existence_check = os.path.exists("model/pytorch_model.bin")
    model_dir = "model/"

    if not model_existence_check:
        text_model_id = "Gustavosta/MagicPrompt-Stable-Diffusion"
        tokenizer_dl = GPT2Tokenizer.from_pretrained(text_model_id)
        model_dl = GPT2LMHeadModel.from_pretrained(text_model_id)
        tokenizer_dl.save_pretrained(model_dir)
        model_dl.save_pretrained(model_dir)

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir, pad_token_id=tokenizer.eos_token_id)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

    seed = randint(100, 1000000)
    set_seed(seed)

    if starting_text == "":
        starting_text: str = input_text.replace("\n", "").lower().capitalize()
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)

    response = pipe(starting_text, max_length=(len(starting_text) + randint(60, 90)), num_return_sequences=4)
    response_list = []
    for x in response:
        resp = x['generated_text'].strip()
        response_list.append(resp)
    for k in range(len(response_list)):
        response_list[k] = re.sub('[^ ]+\\.[^ ]+', '', response_list[k])
        response_list[k] = response_list[k].replace("<", "").replace(">", "")

    if response_list != "":
        return response_list


def generate_images(prompts, prompts_numbers):
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"

    sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    sd_pipe = sd_pipe.to(device)
    images = []
    for number in prompts_numbers:
        image = sd_pipe(prompts[number - 1]).images[0]
        images.append(image)
    return images


if __name__ == '__main__':
    st.title('Welcome To Project Eagle Vision!')
    instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)
    input_text = st.text_input("Your image description")
    if st.button("Generate prompts"):
        if len(input_text) < 1:
            st.write("Incorrect input format. You should write at least one word. Please, try again.")
        else:
            with st.spinner("Wait for it..."):
                prompts = generate_prompts(input_text)
            st.success("Done!")
            st.title("Here are the generated prompts")
            for i in range(len(prompts)):
                st.write(str(i + 1) + ". " + prompts[i])
        st.write("Choose some prompts which you like the most")
        prompts_numbers = st.multiselect("Choose some prompts you like the most",
                                         [1, 2, 3, 4], label_visibility="hidden")
        if st.button("Generate images"):
            st.title("The generated images")
            st.write("Here are the generated images based on your prompts")
            imgs = generate_images(prompts, prompts_numbers)
            col_list = st.columns(len(prompts_numbers))
            for i in range(len(col_list)):
                with col_list[i]:
                    st.image(imgs[i])