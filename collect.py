# Basic example for running MTurk data collection against a Space
# For more information see https://docs.aws.amazon.com/mturk/index.html

import boto3
from boto.mturk.question import ExternalQuestion

from config import MTURK_KEY, MTURK_SECRET
import argparse
from os import path

parser = argparse.ArgumentParser()
parser.add_argument("--mturk_region", default="us-east-1", help="The region for mturk (default: us-east-1)")
parser.add_argument("--space_name", default="huggingface/rlhf-interface", help="Name of the accompanying Hugging Face space (default: huggingface/rlhf-interface)")
parser.add_argument("--num_hits", type=int, default=5, help="The number of HITs.")
parser.add_argument("--num_assignments", type=int, default=1, help="The number of times that the HIT can be accepted and completed.")
parser.add_argument("--live_mode", action="store_true", help="""
    Whether to run in live mode with real turkers. This will charge your account money.
    If you don't use this flag, the HITs will be deployed on the sandbox version of mturk,
    which will not charge your account money.
    """
)
parser.add_argument("--refresh_qualification_test", action="store_true", help="""
    Whether to refresh the qualification test. If you've made edits to the test
    xml files, it is necessary to do this.
    """
)

args = parser.parse_args()

MTURK_URL = f"https://mturk-requester{'' if args.live_mode else '-sandbox'}.{args.mturk_region}.amazonaws.com"

mturk = boto3.client(
    "mturk",
    aws_access_key_id=MTURK_KEY,
    aws_secret_access_key=MTURK_SECRET,
    region_name=args.mturk_region,
    endpoint_url=MTURK_URL,
)

# This is the URL that makes the space embeddable in an mturk iframe
question = ExternalQuestion(f"https://hf.space/embed/{args.space_name}/+?__theme=light",
    frame_height=600
)

qualification_type_id = open("qualification_type_id.txt", "r").read() if path.exists("qualification_type_id.txt") else None
if args.refresh_qualification_test or qualification_type_id is None:
    if qualification_type_id is not None:
        client.delete_qualification_type(
            QualificationTypeId='string'
        )
    response = mturk.create_qualification_type(
        Name='rlhf-qualification',
        Keywords='RLHF qualification',
        Description='Qualification test for RLHF task.',
        QualificationTypeStatus='Active',
        Test=open('qualification_questions.xml', mode='r').read(),
        AnswerKey=open('qualification_answers.xml', mode='r').read(),
        TestDurationInSeconds=3600,
        AutoGranted=False,
    )
    qualification_type_id = response["QualificationType"]["QualificationTypeId"]
    open("qualification_type_id.txt", "w+").write(qualification_type_id)

for i in range(args.num_hits):
    new_hit = mturk.create_hit(
        Title="RLHF HIT",
        Description="Interact with an AI",
        Keywords="chatbot",
        Reward="0.15",
        MaxAssignments=args.num_assignments,
        LifetimeInSeconds=172800,
        AssignmentDurationInSeconds=600,
        AutoApprovalDelayInSeconds=14400,
        Question=question.get_as_xml(),
        QualificationRequirements=[{
            'QualificationTypeId': qualification_type_id,
            'Comparator': 'Exists',
            'RequiredToPreview': False,
            'ActionsGuarded': 'Accept'
        }],
    )

print(
    f"HIT Group Link: https://worker{'' if args.live_mode else 'sandbox'}.mturk.com/mturk/preview?groupId="
    + new_hit["HIT"]["HITGroupId"]
)