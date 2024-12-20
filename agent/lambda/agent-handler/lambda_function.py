import datetime
import difflib
import json
import logging
import os
import time

import boto3
import dateutil.parser
import pdfrw
from boto3.dynamodb.conditions import Key
from chat import Chat
from fsi_agent import FSIAgent
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock

# Create reference to DynamoDB tables and S3 bucket
user_accounts_table_name = os.environ["USER_EXISTING_ACCOUNTS_TABLE"]
s3_artifact_bucket = os.environ["S3_ARTIFACT_BUCKET_NAME"]

# Instantiate boto3 clients and resources
boto3_session = boto3.Session(region_name=os.environ["AWS_REGION"])
dynamodb = boto3.resource("dynamodb", region_name=os.environ["AWS_REGION"])
s3_client = boto3.client(
    "s3",
    region_name=os.environ["AWS_REGION"],
    config=boto3.session.Config(
        signature_version="s3v4",
    ),
)
s3_object = boto3.resource("s3")
bedrock_client = boto3_session.client(service_name="bedrock-runtime")

# --- Lex v2 request/response helpers ---


def elicit_slot(session_attributes, active_contexts, intent, slot_to_elicit, message):
    """
    Constructs a response to elicit a specific Amazon Lex intent slot value from the user during conversation.
    """
    response = {
        "sessionState": {
            "activeContexts": [
                {
                    "name": "intentContext",
                    "contextAttributes": active_contexts,
                    "timeToLive": {"timeToLiveInSeconds": 86400, "turnsToLive": 20},
                }
            ],
            "sessionAttributes": session_attributes,
            "dialogAction": {"type": "ElicitSlot", "slotToElicit": slot_to_elicit},
            "intent": intent,
        },
        "messages": [
            {
                "contentType": "PlainText",
                "content": message,
            }
        ],
    }

    return response


def elicit_intent(intent_request, session_attributes, message):
    """
    Constructs a response to elicit the user's intent during conversation.
    """
    response = {
        "sessionState": {
            "dialogAction": {"type": "ElicitIntent"},
            "sessionAttributes": session_attributes,
        },
        "messages": [
            {"contentType": "PlainText", "content": message},
            {
                "contentType": "ImageResponseCard",
                "imageResponseCard": {
                    "buttons": [
                        {"text": "Emergency Helpline", "value": "Emergency Helpline"},
                        {
                            "text": "Ask Alex Buddy",
                            "value": "What kind of questions can Alex Buddy answer?",
                        },
                    ],
                    "title": "How can I help you?",
                },
            },
        ],
    }

    return response


def delegate(session_attributes, active_contexts, intent, message):
    """
    Delegates the conversation back to the system for handling.
    """
    response = {
        "sessionState": {
            "activeContexts": [
                {
                    "name": "intentContext",
                    "contextAttributes": active_contexts,
                    "timeToLive": {"timeToLiveInSeconds": 86400, "turnsToLive": 20},
                }
            ],
            "sessionAttributes": session_attributes,
            "dialogAction": {
                "type": "Delegate",
            },
            "intent": intent,
        },
        "messages": [{"contentType": "PlainText", "content": message}],
    }

    return response


def try_ex(value):
    """
    Safely access slots dictionary values.
    """
    if value is not None:
        if value["value"]["resolvedValues"]:
            return value["value"]["interpretedValue"]
        elif value["value"]["originalValue"]:
            return value["value"]["originalValue"]
        else:
            return None
    else:
        return None


# --- Intent fulfillment functions ---


def verify_identity(intent_request):
    """
    Performs dialog management and fulfillment for username verification.
    """
    slots = intent_request["sessionState"]["intent"]["slots"]
    username = try_ex(slots["UserName"])

    session_attributes = intent_request["sessionState"].get("sessionAttributes") or {}
    intent = intent_request["sessionState"]["intent"]
    active_contexts = {}

    if username:
        session_attributes["UserName"] = username
        return elicit_intent(
            intent_request,
            session_attributes,
            f"Thank you for confirming your username and PIN, {username}. How are you doing? Is everything going well?",
        )
    else:
        return elicit_slot(
            session_attributes,
            active_contexts,
            intent,
            "UserName",
            "Please provide your username to proceed.",
        )


def emergency_helpline(intent_request):
    """
    Handles mental health emergency helpline requests.
    """
    session_attributes = intent_request["sessionState"].get("sessionAttributes", {})
    intent = intent_request["sessionState"]["intent"]

    # Get the value of the Selection slot
    slots = intent.get("slots", {})
    user_selection = try_ex(slots.get("Selection"))

    if not user_selection:
        # Elicit the Selection slot if it's not provided
        return {
            "sessionState": {
                "dialogAction": {"type": "ElicitSlot", "slotToElicit": "Selection"},
                "intent": intent,
                "sessionAttributes": session_attributes,
            },
            "messages": [
                {
                    "contentType": "PlainText",
                    "content": (
                        "What type of support do you need? Please select one option from below:\n"
                        "1. General Support\n"
                        "2. Suicide Prevention\n"
                        "3. Young People\n"
                        "4. LGBTQ+ Support\n"
                        "5. Urgent Help"
                    ),
                }
            ],
        }

    # Process the user's selection
    if user_selection == "1":
        message = "**General Mental Health Support**:\n- Samaritans: Call 116 123 (24/7)\n- NHS 111: Call 111 (24/7)"
    elif user_selection == "2":
        message = "**Suicide Prevention**:\n- National Suicide Prevention Helpline: Call 0800 689 5652\n- Papyrus HOPELINEUK: Call 0800 068 4141"
    elif user_selection == "3":
        message = "**Support for Young People**:\n- Childline: Call 0800 1111 (24/7)\n- Shout Crisis Text Line: Text 'YM' to 85258"
    elif user_selection == "4":
        message = "**LGBTQ+ Support**:\n- Switchboard: Call 0300 330 0630 (10am-10pm)\n- Text 'SHOUT' to 85258 (24/7)"
    elif user_selection == "5":
        message = "**Urgent Help**:\n- Call 999 for immediate danger\n- Visit A&E for emergency mental health support"
    else:
        message = "I didn’t understand that. Please reply with 1, 2, 3, 4, or 5."

    # Return the message with a "Close" action
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {
                "name": intent["name"],
                "slots": intent["slots"],
                "state": "Fulfilled",
                "confirmationState": intent.get("confirmationState", "None"),
            },
            "sessionAttributes": session_attributes,
        },
        "messages": [
            {
                "contentType": "PlainText",
                "content": message,
            }
        ],
    }


def invoke_agent(prompt, session_id):
    """
    Invokes Amazon Bedrock-powered LangChain agent with 'prompt' input.
    """
    chat = Chat({"Human": prompt}, session_id)
    llm = Bedrock(
        client=bedrock_client,
        model_id="anthropic.claude-v2:1",
        region_name=os.environ["AWS_REGION"],
    )  # anthropic.claude-instant-v1 / anthropic.claude-3-sonnet-20240229-v1:0
    llm.model_kwargs = {"max_tokens_to_sample": 350}
    lex_agent = FSIAgent(llm, chat.memory)

    message = lex_agent.run(input=prompt)

    # summarize response and save in memory
    formatted_prompt = (
        "\n\nHuman: "
        + "Summarize the following within 50 words: "
        + message
        + " \n\nAssistant:"
    )
    conversation = ConversationChain(llm=llm)
    ai_response_recap = conversation.predict(input=formatted_prompt)
    chat.set_memory({"Assistant": ai_response_recap}, session_id)

    return message


def genai_intent(intent_request):
    """
    Performs dialog management and fulfillment for user utterances that do not match defined intents (e.g., FallbackIntent).
    Sends user utterance to the 'invoke_agent' method call.
    """
    session_attributes = intent_request["sessionState"].get("sessionAttributes") or {}
    session_id = intent_request["sessionId"]

    if intent_request["invocationSource"] == "DialogCodeHook":
        prompt = intent_request["inputTranscript"]
        output = invoke_agent(prompt, session_id)
        print("FSI Agent response: " + str(output))

    return elicit_intent(intent_request, session_attributes, output)


# --- Intents ---


def dispatch(intent_request):
    """
    Routes the incoming request based on intent.
    """
    intent_name = intent_request["sessionState"]["intent"]["name"]

    if intent_name == "VerifyIdentity":
        return verify_identity(intent_request)
    elif intent_name == "Emergencyhelpline":
        return emergency_helpline(intent_request)
    else:
        return genai_intent(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


# --- Main handler ---


def handler(event, context):
    """
    Invoked when the user provides an utterance that maps to a Lex bot intent.
    The JSON body of the user request is provided in the event slot.
    """
    os.environ["TZ"] = "America/New_York"
    time.tzset()

    return dispatch(event)
