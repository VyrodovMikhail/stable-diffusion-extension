import torch
from diffusers import StableDiffusionPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed, pipelines
from random import randint
import re
import streamlit as st

MODEL_SEED = randint(100, 1000000)


def load_prompts_pipe():
    model_dir = "model/"

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir, pad_token_id=tokenizer.eos_token_id)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

    set_seed(MODEL_SEED)

    return pipe


def load_images_pipe():
    model_id = "CompVis/stable-diffusion-v1-4"
    if torch.cuda.is_available():
        sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
        sd_pipe = sd_pipe.to("cuda")
    else:
        sd_pipe = StableDiffusionPipeline.from_pretrained(model_id)
        sd_pipe = sd_pipe.to("cpu")

    return sd_pipe


def generate_prompts(pipe, max_input_length, num_return_prompts, starting_text):
    if starting_text == "":
        starting_text: str = input_text.replace("\n", "").lower().capitalize()
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)

    response = pipe(starting_text, max_length=max_input_length, num_return_sequences=num_return_prompts)
    response_list = []
    for x in response:
        resp = x['generated_text'].strip()
        response_list.append(resp)
    for k in range(len(response_list)):
        response_list[k] = re.sub('[^ ]+\\.[^ ]+', '', response_list[k])
        response_list[k] = response_list[k].replace("<", "").replace(">", "")

    if response_list != "":
        return response_list


@st.cache(hash_funcs={StableDiffusionPipeline: lambda _: None}, show_spinner=False)
def generate_images(pipe, prompts, prompts_numbers):
    images = []
    for number in prompts_numbers:
        image = pipe(prompts[number - 1]).images[0]
        images.append(image)
    return images


@st.cache(hash_funcs={pipelines.text_generation.TextGenerationPipeline: lambda _: None})
def get_prompts(pipe, max_input_length, num_return_prompts, input_text):
    prompts = generate_prompts(pipe, max_input_length, num_return_prompts, input_text)
    return prompts


if __name__ == '__main__':

    if not "prompts_model_loaded" in st.session_state:
        st.session_state.prompts_model_loaded = False

    if not "prompts_pipe" in st.session_state:
        st.session_state.prompts_pipe = None

    if not "images_model_loaded" in st.session_state:
        st.session_state.images_model_loaded = False

    if not "images_pipe" in st.session_state:
        st.session_state.images_pipe = None

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
        if not len(input_text):
            st.write("Incorrect input format. You should write at least one word. Please, try again.")
        else:
            if not st.session_state.prompts_model_loaded:
                st.session_state.prompts_pipe = load_prompts_pipe()
                st.session_state.prompts_model_loaded = True
            generated_prompts = get_prompts(st.session_state.prompts_pipe, 90, 4, input_text)

            st.title("Here are the generated prompts")
            for i in range(len(generated_prompts)):
                st.write(str(i + 1) + ". " + generated_prompts[i])

            if not "prompts_numbers" in st.session_state:
                st.session_state.prompts_numbers = []

            available_numbers = []
            for i in range(len(generated_prompts)):
                available_numbers.append(i + 1)

            prompts_numbers = st.multiselect("Choose some prompts you like the most",
                                             available_numbers, label_visibility="hidden")
            st.session_state.prompts_numbers = prompts_numbers

            images_button = st.button("Generate images")
            if images_button and len(st.session_state.prompts_numbers) > 0:
                if not st.session_state.images_model_loaded:
                    st.session_state.images_pipe = load_images_pipe()
                    st.session_state.images_model_loaded = True
                with st.spinner("Generating images..."):
                    images = generate_images(st.session_state.images_pipe,
                                             generated_prompts, st.session_state.prompts_numbers)
                st.success('Done!')

                st.title("The generated images")
                st.write("Here are the generated images based on your prompts")
                col_list = st.columns(len(st.session_state.prompts_numbers))

                for i in range(len(col_list)):
                    with col_list[i]:
                        st.image(images[i])