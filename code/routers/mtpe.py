import os
import random

# from pathlib import Path
from code.get_models import get_model_id
from code.models.mtpe import ScoredTranslation, Translation
from glob import glob
from typing import List

# from comet import download_model, load_from_checkpoint
# from huggingface_hub import snapshot_download
from comet import load_from_checkpoint
from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException  # , status
from huggingface_hub import login

# from comet import download_model, load_from_checkpoint
# from huggingface_hub import snapshot_download


load_dotenv()
AUTH_KEY = os.environ.get("AUTH_KEY")
custom_hf_cache_dpath = os.getenv("HF_HOME")
# HF_HOME = os.getenv("HF_HOME")
# HF_HOME = os.environ["HF_HOME"]
TOKEN = os.environ["HUGGINGFACE_TOKEN"]
login(TOKEN)  # huggingface-cli login --token $HUGGINGFACE_TOKEN
# login(token=TOKEN, add_to_git_credential=True)
clean_up_tokenization_spaces = True


router = APIRouter()


def get_model_ckpt_fpath(model_id):
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model specified in query parameter, model '{model_id}' does not exist on Hugging Face.",
        )

    model_dir = f"models--{model_id.replace('/', '--')}"

    # Use glob to find the file matching the pattern
    return glob(
        os.path.join(f"{custom_hf_cache_dpath}/{model_dir}", "**", "model.ckpt"),
        recursive=True,
    )[0]


def produce_scores(model_ckpt, data):
    data_dict = [{"src": obj.src, "mt": obj.mt} for obj in data]
    # model_output = model.predict(data, batch_size=8, gpus=1)
    # print(f"{model_output.scores=}")  # sentence-level scores
    # print(f"{model_output.system_score=}")  # average score
    return model_ckpt.predict(data_dict, batch_size=8, gpus=1).scores


def produce_scores_mock(data):
    data_dict = [{"src": obj.src, "mt": obj.mt} for obj in data]
    return [round(random.uniform(0.5, 1), 15) for x in range(len(data_dict))]


def add_scores_to_data(data, scores):
    # scores = produce_scores(data)
    return [
        ScoredTranslation(
            **translation.dict(),  # Unpack the Translation instance data
            score=score,  # Add the new score value
        )
        for translation, score in zip(data, scores)
    ]


@router.get("/scores")
async def get_scores(
    translations: List[Translation],
    model: str = "Unbabel/wmt22-cometkiwi-da",
    mode: str = "mock",
    authorization: str = Header(None),  # extracts 'Authorization' header
):
    if not authorization or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if mode == "mock":
        scores = produce_scores_mock(translations)
    else:
        model_id = get_model_id(model)
        model_fpath = get_model_ckpt_fpath(model_id)
        model_ckpt = load_from_checkpoint(model_fpath)
        scores = produce_scores(model_ckpt, translations)

    # model_output_scores = [0.3048415184020996, 0.23436091840267181, 0.6128204464912415]
    # model_output_system_score = 0.38400762776533764

    return {
        "data": add_scores_to_data(translations, scores),
        "model_output_system_score": sum(scores) / len(scores),
    }