import json
import logging
import os
from collections import OrderedDict

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm

from src.classification_models.quantized_llama_based_models import (
    LLaMABasedQuantizedModel,
)
from src.experiments_pipelines.chatbot import ChatBotLLM
from src.process_docanno_data import process_data
from src.utils import read_jsonl

CHAIN_OF_THOUGHT = {
    0: """
    {instruction_begin}

    Text: "{example_input}"

    Determine whether the following sentence contains a fallacy or not:
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    
    """,
    1: """
    {instruction_begin}

    Text: "{example_input}"

    Based on the above text, identify the fallacy (if any) in the following sentence. If a fallacy is present, specify the type(s) of fallacy without providing explanations. The possible types of fallacy are:
    - appeal to emotion
    - fallacy of logic
    - fallacy of credibility    
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
        
    """,
    2: """
    You are a helpful assistant which is an expert in detecting logical fallacies!
    Please identify any and all logical fallacies present in the following text.\n
    Let's think step by step by going through the sentences and checking them for every single fallacy:

    "{example_input}"

    {instruction_begin}
    Based on the above text, identify the fallacy (if any) in the following sentence. If a fallacy is present, specify the type(s) of fallacy without providing explanations. The possible types of fallacy are:
    - appeal to positive emotion
    - appeal to anger
    - appeal to fear
    - appeal to pity
    - appeal to ridicule
    - appeal to worse problems
    - causal oversimplification
    - circular reasoning
    - equivocation
    - false analogy
    - false causality
    - false dilemma
    - hasty generalization
    - slippery slope
    - straw man
    - fallacy of division
    - ad hominem
    - ad populum
    - appeal to (false) authority
    - appeal to nature
    - appeal to tradition
    - guilt by association
    - tu quoque
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    """,
}


THREAD_OF_THOUGHT_ZERO_SHOT = {
    0: """
    {instruction_begin}

    Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.

    Text: "{example_input}"

    Based on the above text, determine whether the following sentence is part of a fallacious argument or not:
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    
    """,
    1: """
    {instruction_begin}

    Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.

    Text: "{example_input}"

    Based on the above text, determine whether the following sentence is part of a fallacious argument or not. If it is, indicate the type(s) of fallacy without providing explanations. The potential types of fallacy include:
    - appeal to emotion
    - fallacy of logic
    - fallacy of credibility    
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
        
    """,
    2: """
    As a content reviewer, I provide multiple fallacies which might occur in the text.
    You need to identify the fallacies, if any, in the text.
    Text: "{example_input}"
    {instruction_begin}
    Walk me through this context in manageable parts step by step,
    summarizing and analyzing as we go.
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Answer:
    
    """,
}

THREAD_OF_THOUGHT_FEW_SHOT = {
    0: """
    {instruction_begin}

    Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.

    Text: "{example_input}"

    Based on the above text, determine whether the following sentence is part of a fallacious argument or not:
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
    
    """,
    1: """
    {instruction_begin}

    Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.

    Text: "{example_input}"

    Based on the above text, determine whether the following sentence is part of a fallacious argument or not. If it is, indicate the type(s) of fallacy without providing explanations. The potential types of fallacy include:
    - appeal to emotion
    - fallacy of logic
    - fallacy of credibility    
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Output:
        
    """,
    2: """
    As a content reviewer, I provide multiple fallacies which might occur in the text.
    You need to identify the fallacies, if any, in the text.
    Slippery slope:\n\nIf we ban assault rifles, next they'll ban all guns, and before you know it, we'll be living in a police state.,
    Hasty generalization:\n\nI met two people from New York City, and both were rude. Therefore, all New Yorkers are rude.,
    False analogy:\n\nComparing buying a car to choosing a spouse is a false analogy because the two situations are vastly different.,
    Guilt by association:\n\nYou shouldn't listen to John's opinion on politics; he's friends with several criminals.,
    Causal oversimplification:\n\nThe rise in crime rates is due to violent video games; therefore, banning them will solve the problem.,
    Ad populum:\n\nEveryone is buying the new smartphone, so it must be the best one on the market.,
    Circular reasoning:\n\nThe Bible is the word of God because God says it is in the Bible.,
    Appeal to fear:\n\nIf you don't vote for me, your taxes will skyrocket and crime will run rampant.,
    False causality:\n\nEvery time I wash my car, it rains. Therefore, washing my car causes rain.,
    Fallacy of division:\n\nThat company is profitable, so all of its employees must be well-paid.,
    Appeal to ridicule:\n\nYou shouldn't listen to Mary's argument about climate change; she believes the Earth is flat!,
    Appeal to worse problems:\n\nWhy are we worrying about pollution when there are children starving in the world?,
    Abusive ad hominem:\n\nYou can't trust anything Tom says; he's just a loser.,
    Appeal to false authority:\n\nMy doctor says that smoking is good for you, so it must be true.,
    Appeal to positive emotion:\n\nIf you care about the environment, you'll buy this eco-friendly product.,
    Appeal to nature:\n\nNatural remedies are always better than pharmaceutical drugs.,
    Equivocation:\n\nThe sign said 'Fine for parking here,' so I thought it would be fine to park there.,
    Appeal to anger:\n\nYou wouldn't want to be a coward, would you? Then you better join the military!,
    False dilemma:\n\nEither we ban all guns or we'll continue to have mass shootings.,
    Straw man:\n\nOpponents of gun control argue that we should give guns to everyone, including criminals and children.,
    Appeal to tradition:\n\nWe should continue to use fossil fuels because that's how we've always done it.,
    Tu quoque:\n\nYou say I shouldn't smoke, but I've seen you smoke before!
    Text: "{example_input}"
    {instruction_begin}
    Walk me through this context in manageable parts step by step,
    summarizing and analyzing as we go.
    
    Sentence: "{sentence_input}" {instruction_end}
    
    Answer:
    
    """,
}

# THREAD_OF_THOUGHT_ZERO_SHOT_ADV = {
#     0: """
#     {instruction_begin}

#     Definitions:
#     - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
#     - A fallacious argument is an argument where the premises do not entail the conclusion.

#     Text: "{example_input}"

#     Based on the above text, determine whether the following sentence is part of a fallacious argument or not:
    
#     Sentence: "{sentence_input}" {instruction_end}
    
#     Output:
    
#     """,
#     1: """
#     {instruction_begin}

#     Definitions:
#     - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
#     - A fallacious argument is an argument where the premises do not entail the conclusion.

#     Text: "{example_input}"

#     Based on the above text, determine whether the following sentence is part of a fallacious argument or not. If it is, indicate the type(s) of fallacy without providing explanations. The potential types of fallacy include:
#     - appeal to emotion
#     - fallacy of logic
#     - fallacy of credibility    
    
#     Sentence: "{sentence_input}" {instruction_end}
    
#     Output:
        
#     """,
#     2: """
#     As a content reviewer, I provide multiple fallacies which might occur in the text.
#     You need to identify the fallacies, if any, in the text.
#     Slippery slope:\n\nIf we ban assault rifles, next they'll ban all guns, and before you know it, we'll be living in a police state.,
#     Hasty generalization:\n\nI met two people from New York City, and both were rude. Therefore, all New Yorkers are rude.,
#     False analogy:\n\nComparing buying a car to choosing a spouse is a false analogy because the two situations are vastly different.,
#     Guilt by association:\n\nYou shouldn't listen to John's opinion on politics; he's friends with several criminals.,
#     Causal oversimplification:\n\nThe rise in crime rates is due to violent video games; therefore, banning them will solve the problem.,
#     Ad populum:\n\nEveryone is buying the new smartphone, so it must be the best one on the market.,
#     Circular reasoning:\n\nThe Bible is the word of God because God says it is in the Bible.,
#     Appeal to fear:\n\nIf you don't vote for me, your taxes will skyrocket and crime will run rampant.,
#     False causality:\n\nEvery time I wash my car, it rains. Therefore, washing my car causes rain.,
#     Fallacy of division:\n\nThat company is profitable, so all of its employees must be well-paid.,
#     Appeal to ridicule:\n\nYou shouldn't listen to Mary's argument about climate change; she believes the Earth is flat!,
#     Appeal to worse problems:\n\nWhy are we worrying about pollution when there are children starving in the world?,
#     Abusive ad hominem:\n\nYou can't trust anything Tom says; he's just a loser.,
#     Appeal to false authority:\n\nMy doctor says that smoking is good for you, so it must be true.,
#     Appeal to positive emotion:\n\nIf you care about the environment, you'll buy this eco-friendly product.,
#     Appeal to nature:\n\nNatural remedies are always better than pharmaceutical drugs.,
#     Equivocation:\n\nThe sign said 'Fine for parking here,' so I thought it would be fine to park there.,
#     Appeal to anger:\n\nYou wouldn't want to be a coward, would you? Then you better join the military!,
#     False dilemma:\n\nEither we ban all guns or we'll continue to have mass shootings.,
#     Straw man:\n\nOpponents of gun control argue that we should give guns to everyone, including criminals and children.,
#     Appeal to tradition:\n\nWe should continue to use fossil fuels because that's how we've always done it.,
#     Tu quoque:\n\nYou say I shouldn't smoke, but I've seen you smoke before!
#     Text: "{example_input}"
#     {instruction_begin}
#     Walk me through this context in manageable parts step by step,
#     summarizing and analyzing as we go.
    
#     Sentence: "{sentence_input}" {instruction_end}
    
#     Answer:
    
#     """,
# }

def zero_or_few_shots_pipeline(
    model: LLaMABasedQuantizedModel,
    dataset_path: str = None,
    prompt: str = None,
    prediction_path: str = None,
    level: int = 0,
):
    logger = logging.getLogger("MafaldaLogger")

    if prompt == "CoT":
        template = CHAIN_OF_THOUGHT[level]
    elif prompt == "ToT-zero":
        template = THREAD_OF_THOUGHT_ZERO_SHOT[level]
    elif prompt == "ToT-few":
        template = THREAD_OF_THOUGHT_FEW_SHOT[level]
    elif prompt == "ToT-zero-adv":
        raise ValueError("ToT-zero-adv is not supported.")
        # template = THREAD_OF_THOUGHT_ZERO_SHOT_ADV[level]
    elif prompt == "ToT-few-adv":
        raise ValueError("ToT-few-adv is not supported.")
        # template = THREAD_OF_THOUGHT_ZERO_SHOT_ADV[level]

    prompt = PromptTemplate(
            input_variables=[
                "example_input",
                "sentence_input",
                "instruction_begin",
                "instruction_end",
            ],
            template=template,
        )

    chatbot_model = ChatBotLLM(model=model)
    if model.model_name == "gpt-3.5":
        chatbot_model.max_length = 1024
    chatbot_chain = LLMChain(llm=chatbot_model, prompt=prompt)

    data = read_jsonl(dataset_path)
    processed_data = process_data(data)
    assert len(data) == len(
        processed_data
    ), f"Data length mismatch: {len(data)} != {len(processed_data)}"

    # Check already processed examples
    already_processed = set()
    if os.path.exists(prediction_path):
        with open(prediction_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    already_processed.add(entry["text"])
                except json.JSONDecodeError:
                    # Handle improperly formatted last line
                    f.seek(0)
                    all_lines = f.readlines()
                    with open(prediction_path, "w") as fw:
                        fw.writelines(all_lines[:-1])

    # Ensure the prediction_path directory exists
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    with open(prediction_path, "a") as f:
        for example, processed_example in tqdm(
            zip(data, processed_data), total=len(data)
        ):
            if example["text"] in already_processed:
                logger.info(f"Skipping already processed example: {example['text']}")
                continue

            logger.info(example["text"])
            pred_outputs = OrderedDict()
            for s in processed_example:
                logger.info(s)
                input = {
                    "example_input": example["text"],
                    "sentence_input": s,
                    "instruction_begin": model.instruction_begin,
                    "instruction_end": model.instruction_end
                }
                output = chatbot_chain.invoke(input)

                logger.info(output)
                pred_outputs[s] = output
            json_line = json.dumps(
                {
                    "text": example["text"],
                    "prediction": pred_outputs,
                }
            )
            f.write(json_line + "\n")
