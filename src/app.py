import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed
from random import randint
import re
import streamlit as st


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
    if torch.cuda.is_available():
        sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
        sd_pipe = sd_pipe.to("cuda")
    else:
        sd_pipe = StableDiffusionPipeline.from_pretrained(model_id)
        sd_pipe = sd_pipe.to("cpu")
    images = []
    for number in prompts_numbers:
        image = sd_pipe(prompts[number - 1]).images[0]
        images.append(image)
    return images


@st.cache()
def generate_input_text_flag(input_text):
    if len(input_text) <= 1:
        return True
    return False


@st.cache(suppress_st_warning=True)
def show_prompts(input_text):
    prompts = generate_prompts(input_text)
    return prompts


@st.cache()
def get_images(prompts, prompts_numbers):
    imgs = generate_images(prompts, prompts_numbers)
    return imgs


if __name__ == '__main__':
    st.title('Welcome To Stable Diffusion extension')
    instructions = """
        You should upload a small text or even one word that describes
        images you want to generate. After pressing \"Generate Prompts\" button
        you receive 4 image descriptions. They can be a little unclear because of
        they are specifically generated for the Stable Diffusion model.
        Choose some of them which you like the most. After pressing button \"Generate Images\"
        you receive images corresponding to your chosen prompts numbers
        in order which you've picked them.
        """
    st.write(instructions)
    input_text = st.text_input("Your image description")
    if not "prompts_button" in st.session_state:
        st.session_state.prompts_button = False

    prompts_button = st.button("Generate prompts")
    if st.session_state.prompts_button or prompts_button:
        st.session_state.prompts_button = True
        flag = generate_input_text_flag(input_text)
        if flag:
            st.write("Incorrect input format. You should write at least one word. Please, try again.")
        else:
            generated_prompts = show_prompts(input_text)
            st.title("Here are the generated prompts")
            for i in range(len(generated_prompts)):
                st.write(str(i + 1) + ". " + generated_prompts[i])

            if not "prompts_numbers" in st.session_state:
                st.session_state.prompts_numbers = []

            prompts_numbers = st.multiselect("Choose some prompts you like the most",
                                             [1, 2, 3, 4], label_visibility="hidden")

            st.session_state.prompts_numbers = prompts_numbers

            images_button = st.button("Generate images")

            if images_button and len(st.session_state.prompts_numbers) > 0:
                images = get_images(generated_prompts, st.session_state.prompts_numbers)
                st.title("The generated images")
                st.write("Here are the generated images based on your prompts")
                col_list = st.columns(len(st.session_state.prompts_numbers))
                for i in range(len(col_list)):
                    with col_list[i]:
                        st.image(images[i])