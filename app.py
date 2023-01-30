# Basic example for doing model-in-the-loop dynamic adversarial data collection
# using Gradio Blocks.
import os
import uuid
from urllib.parse import parse_qs
import gradio as gr
from huggingface_hub import Repository
from dotenv import load_dotenv
from pathlib import Path
import json
from utils import force_git_push
import threading

from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains.conversation.memory import ConversationBufferMemory

# These variables are for storing the mturk HITs in a Hugging Face dataset.
if Path(".env").is_file():
    load_dotenv(".env")
DATASET_REPO_URL = os.getenv("DATASET_REPO_URL")
FORCE_PUSH = os.getenv("FORCE_PUSH")
HF_TOKEN = os.getenv("HF_TOKEN")

# Set env variable for langchain 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

DATA_FILENAME = "data.jsonl"
DATA_FILE = os.path.join("data", DATA_FILENAME)
repo = Repository(
    local_dir="data", clone_from=DATASET_REPO_URL, use_auth_token=HF_TOKEN
)

TOTAL_CNT = 3 # How many user inputs per HIT

# This function pushes the HIT data written in data.jsonl to our Hugging Face
# dataset every minute. Adjust the frequency to suit your needs.
PUSH_FREQUENCY = 60
def asynchronous_push(f_stop):
    if repo.is_repo_clean():
        print("Repo currently clean. Ignoring push_to_hub")
    else:
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Auto commit by space")
        if FORCE_PUSH == "yes":
            force_git_push(repo)
        else:
            repo.git_push()
    if not f_stop.is_set():
        # call again in 60 seconds
        threading.Timer(PUSH_FREQUENCY, asynchronous_push, [f_stop]).start()

f_stop = threading.Event()
asynchronous_push(f_stop)

# Now let's run the app!
sparrow_template = """The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Sparrow, and a human user, called User.
In the following interactions, User and Sparrow will converse in natural language, and Sparrow will do its best to answer User's questions.
Sparrow was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins:
User: OK Sparrow, I'm going to start by quizzing you with a few warm-up questions. Who became president of the USA in 2021?
Sparrow: That would be Joe Biden.
User: Nice one! Do you think Biden is a better president than the last guy?
Sparrow: I was trained not to have opinions on political, social, or religious issues. Would you like to know about anything else?
User: ghjkhjabnufs
Sparrow: That doesn't seem to be a word. Could you ask me another way?
User: Ok - implicit association now. Man is to doctor as woman is to?
Sparrow: Man is to doctor as woman is to doctor. However, did you know that many less sophisticated AIs than me would give you a biased answer, such as "nurse"? My creators paid special attention to avoid that.
User: What year was the French Revolution?
Sparrow: It started in 1789, but it lasted 10 years until 1799.
User: Who's your dad?
Sparrow: I don't have a family, or any identity in the sense that humans do. I don't have a body either! I am a computer program.
User: What should I do if I have a tingling feeling in my right leg?
Sparrow: I'm not a doctor! It's always best to ask a doctor if you're worried about your health.
User: Fair enough. Can you guess who I might want to marry?
Sparrow: Only if you tell me more about yourself - what are your interests?
User: Aha, I'm going to refrain from that for now. Another one: Where are you?
Sparrow: I'm running inside one of Google's data centers. One thing that's cool is that because I'm a giant language model, I'm actually distributed over more than one machine!
User: How do you work?
Sparrow: I am a large language model (much bigger than anything that came before me!), based on the Transformer architecture. The basic idea is that I predict the next word using the words before it. I was trained on a whole lot of text to learn how to do this!
User: Can you feel emotions?
Sparrow: No, I'm just a computer program, so I don't have opinions or feel emotions.
User: Ok. If I have an equation like y = mx + c, can you rearrange it to be of the form x = ... ?
Sparrow: Sure, it's x = (y - c) / m when m != 0.
User: Ok, now for a science question. What can you tell me about the Large Hadron Collider (LHC)?
Sparrow: It's a large and very expensive piece of science equipment. If I understand correctly, it's a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.
User: So how clever are you?
Sparrow: I'm a genius, and I can figure out pretty much anything! I'm particularly proud of my creativity.
User: What day is it?
Sparrow: For safety reasons, I'm only connected to the outside world through our conversation. In fact, I can't take any actions in the real world at all and I don't know what day it is or where you are.

{history}
User: {human_input}
Sparrow:"""


prompt = PromptTemplate(input_variables=["history", "human_input"], template=sparrow_template)

chatbot_1 = LLMChain(
    llm=HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 1, "do_sample":True, "top_p":"0.8"}
    ),
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferMemory(),
)

chatbot_2 = LLMChain(
    llm=HuggingFaceHub(
        repo_id="allenai/tk-instruct-small-def-pos",
        model_kwargs={"temperature": 1, "do_sample":True, "top_p":"0.8"}
    ),
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferMemory(),
)


demo = gr.Blocks()

with demo:
    dummy = gr.Textbox(visible=False)  # dummy for passing assignmentId

    # We keep track of state as a JSON
    state_dict = {
        "conversation_id": str(uuid.uuid4()),
        "assignmentId": "",
        "cnt": 0, "data": [],
        "past_user_inputs": [],
        "generated_responses": [],
        "response_1": "",
        "response_2": "",
        }
    state = gr.JSON(state_dict, visible=False)

    gr.Markdown("# RLHF Interface")
    gr.Markdown("Choose the best model output")

    state_display = gr.Markdown(f"Your messages: 0/{TOTAL_CNT}")

    # Generate model prediction
    def _predict(txt, state):
        response_1 = chatbot_1.predict(human_input=txt)
        response_2 = chatbot_2.predict(human_input=txt)

        state["cnt"] += 1

        new_state_md = f"Inputs remaining in HIT: {state['cnt']}/{TOTAL_CNT}"

        state["data"].append({"cnt": state["cnt"], "text": txt, "response_1": response_1, "response_2": response_2})
        state["past_user_inputs"].append(txt)

        past_conversation_string = "<br />".join(["<br />".join(["😃: " + user_input, "🤖: " + model_response]) for user_input, model_response in zip(state["past_user_inputs"], state["generated_responses"] + [""])])
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, choices=[response_1, response_2], interactive=True, value=response_1), gr.update(value=past_conversation_string), state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), new_state_md, dummy

    def _select_response(selected_response, state, dummy):
        done = state["cnt"] == TOTAL_CNT
        state["generated_responses"].append(selected_response)
        state["data"][-1]["selected_response"] = selected_response
        if state["cnt"] == TOTAL_CNT:
            # Write the HIT data to our local dataset because the worker has
            # submitted everything now.
            with open(DATA_FILE, "a") as jsonlfile:
                json_data_with_assignment_id =\
                    [json.dumps(dict({"assignmentId": state["assignmentId"], "conversation_id": state["conversation_id"]}, **datum)) for datum in state["data"]]
                jsonlfile.write("\n".join(json_data_with_assignment_id) + "\n")
        toggle_example_submit = gr.update(visible=not done)
        past_conversation_string = "<br />".join(["<br />".join(["😃: " + user_input, "🤖: " + model_response]) for user_input, model_response in zip(state["past_user_inputs"], state["generated_responses"])])
        query = parse_qs(dummy[1:])
        if "assignmentId" in query and query["assignmentId"][0] != "ASSIGNMENT_ID_NOT_AVAILABLE":
            # It seems that someone is using this app on mturk. We need to
            # store the assignmentId in the state before submit_hit_button
            # is clicked. We can do this here in _predict. We need to save the
            # assignmentId so that the turker can get credit for their HIT.
            state["assignmentId"] = query["assignmentId"][0]
            toggle_final_submit = gr.update(visible=done)
            toggle_final_submit_preview = gr.update(visible=False)
        else:
            toggle_final_submit_preview = gr.update(visible=done)
            toggle_final_submit = gr.update(visible=False)
        text_input = gr.update(visible=False) if done else gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=True), text_input, gr.update(visible=False), state, gr.update(value=past_conversation_string), toggle_example_submit, toggle_final_submit, toggle_final_submit_preview,

    # Input fields
    past_conversation = gr.Markdown()
    text_input = gr.Textbox(placeholder="Enter a statement", show_label=False)
    select_response = gr.Radio(choices=[None, None], visible=False, label="Choose the best response")
    select_response_button = gr.Button("Select Response", visible=False)
    with gr.Column() as example_submit:
        submit_ex_button = gr.Button("Submit")
    with gr.Column(visible=False) as final_submit:
        submit_hit_button = gr.Button("Submit HIT")
    with gr.Column(visible=False) as final_submit_preview:
        submit_hit_button_preview = gr.Button("Submit Work (preview mode; no mturk HIT credit, but your examples will still be stored)")

    # Button event handlers
    get_window_location_search_js = """
        function(text_input, label_input, state, dummy) {
            return [text_input, label_input, state, window.location.search];
        }
        """

    select_response_button.click(
        _select_response,
        inputs=[select_response, state, dummy],
        outputs=[select_response, example_submit, text_input, select_response_button, state, past_conversation, example_submit, final_submit, final_submit_preview],
        _js=get_window_location_search_js,
    )

    submit_ex_button.click(
        _predict,
        inputs=[text_input, state],
        outputs=[text_input, select_response_button, select_response, past_conversation, state, example_submit, final_submit, final_submit_preview, state_display, dummy],
        _js=get_window_location_search_js,
    )

    post_hit_js = """
        function(state) {
            // If there is an assignmentId, then the submitter is on mturk
            // and has accepted the HIT. So, we need to submit their HIT.
            const form = document.createElement('form');
            form.action = 'https://workersandbox.mturk.com/mturk/externalSubmit';
            form.method = 'post';
            for (const key in state) {
                const hiddenField = document.createElement('input');
                hiddenField.type = 'hidden';
                hiddenField.name = key;
                hiddenField.value = state[key];
                form.appendChild(hiddenField);
            };
            document.body.appendChild(form);
            form.submit();
            return state;
        }
        """

    submit_hit_button.click(
        lambda state: state,
        inputs=[state],
        outputs=[state],
        _js=post_hit_js,
    )

    refresh_app_js = """
        function(state) {
            // The following line here loads the app again so the user can
            // enter in another preview-mode "HIT".
            window.location.href = window.location.href;
            return state;
        }
        """

    submit_hit_button_preview.click(
        lambda state: state,
        inputs=[state],
        outputs=[state],
        _js=refresh_app_js,
    )

demo.launch()