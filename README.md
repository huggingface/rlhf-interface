---
title: RLHF
emoji: 🏢
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 3.1
app_file: app.py
pinned: false
---

A basic example of an RLHF interface with a Gradio app.

**Instructions for someone to use for their own project:**

*Setting up the Space*
1. Clone this repo and deploy it on your own Hugging Face space.
2. Add the following secrets to your space:
   - `HF_TOKEN`: One of your Hugging Face tokens.
   - `DATASET_REPO_URL`: The url to an empty dataset that you created the hub. It
    can be a private or public dataset.
   - `FORCE_PUSH`: "yes"
   When you run this space on mturk and when people visit your space on
   huggingface.co, the app will use your token to automatically store new HITs
   in your dataset. Setting `FORCE_PUSH` to "yes" ensures that your repo will
   force push changes to the dataset during data collection. Otherwise,
   accidental manual changes to your dataset could result in your space gettin
   merge conflicts as it automatically tries to push the dataset to the hub. For
   local development, add these three keys to a `.env` file, and consider setting
   `FORCE_PUSH` to "no".
*Running Data Collection*
1. On your local repo that you pulled, create a copy of `config.py.example`,
   just called `config.py`. Now, put keys from your AWS account in `config.py`.
   These keys should be for an AWS account that has the
   AmazonMechanicalTurkFullAccess permission. You also need to
   create an mturk requestor account associated with your AWS account.
2. Run `python collect.py` locally.

*Profit*
Now, you should be watching hits come into your Hugging Face dataset
automatically!

*Tips and Tricks*
- Use caution while doing local development of your space and
simultaneously running it on mturk. Consider setting `FORCE_PUSH` to "no" in
your local `.env` file.
- huggingface spaces have limited computational resources and memory. If you
run too many HITs and/or assignments at once, then you could encounter issues.
You could also encounter issues if you are trying to create a dataset that is
very large. Check the log of your space for any errors that could be happening.

