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
    You are a helpful assistant who is an expert in detecting logical fallacies!
    Please identify any and all logical fallacies present in the following text.\n

    "{example_input}"

    {instruction_begin}
    Let's think step by step by going through the sentences and checking them for every single fallacy:
    Identify if this text contains the fallacy 'slippery slope'. Here is an example for a slippery slope: Person A: “I think we should lower the legal drinking age.” Person B: “No, if we do that, we’ll have ten-year-olds getting drunk in bars!”
    Identify if this text contains the fallacy 'hasty generalization'. Here is an example for a hasty generalization: You have a transit flight via Frankfurt Airport, Germany. On the way to your gate, several passengers hastily bump into you without even apologizing. You conclude that “Germans are so rude!”
    Identify if this text contains the fallacy 'false analogy'. Here is an example for a false analogy: Love is like a spring shower. It brings refreshment to a person's body. (Does it also sometimes lead to thunderstorms and being hit by lightning?)
    Identify if this text contains the fallacy 'guilt by association'. Here is an example for guilt by association: This form of the argument is as follows: Group A makes a particular claim. Group B, which is currently viewed negatively by some, makes the same claim as Group A. Therefore, Group A is viewed as associated with Group B, and is now also viewed negatively. An example of this fallacy would be "My opponent for office just received an endorsement from the Puppy Haters Association. Is that the sort of person you would want to vote for?"
    Identify if this text contains the fallacy 'causal oversimplification'. Here is an example for causal oversimplification: 'I water the garden every day, which is why it grows so well.' Watering the garden is certainly one cause of its growth, but the soil quality, amount of sunlight, and number of pests in the area are also major causes.
    Identify if this text contains the fallacy 'ad populum'. Here is an example for ad populum: You’re at a bookstore browsing for books with a friend. Although you are an avid sci-fi reader, your friend picks up a memoir and tells you that you should read the book because it’s a bestseller.
    Identify if this text contains the fallacy 'circular reasoning'. Here is an example for circular reasoning: Parent: “It’s time to go to bed.” Child: “Why?” Parent: “Because this is your bedtime.”
    Identify if this text contains the fallacy 'appeal to fear'. Here is an example for appeal to fear: Either P or Q is true. Q is frightening. Therefore, P is true.
    Identify if this text contains the fallacy 'false causality'. Here is an example for false causality: Every time I bring my umbrella with me, it rains. Clearly, if I leave it at home, there will be sunshine!
    Identify if this text contains the fallacy 'fallacy of division'. Here is an example for fallacy of division: The second grade in Jefferson Elementary eats a lot of ice cream Carlos is a second-grader in Jefferson Elementary Therefore, Carlos eats a lot of ice cream
    Identify if this text contains the fallacy 'appeal to ridicule'. Here is an example for appeal to ridicule: Person A: At one time in prehistory, the continents were fused together into a single supercontinent, which we call Pangaea. Person B: Yes, I definitely believe that hundreds of millions of years ago, some laser cut through the Earth and broke apart a giant landmass into many different pieces.
    Identify if this text contains the fallacy 'appeal to worse problems'. Here is an example for appeal to worse problems: Everyone should wear seatbelts. We should also wear bibs and sleep in a bassinet.
    Identify if this text contains the fallacy 'abusive ad hominem'. Here is an example for abusive ad hominem: who is going to vote for a person looking like this?
    Identify if this text contains the fallacy 'appeal to false authority'. Here is an example for appeal to false authority: My favorite actor, who starred in that movie about a virus that turns people into zombies, said in an interview that genetically modified crops caused COVID-19. So I think that’s what really happened
    Identify if this text contains the fallacy 'appeal to positive emotion'. Here is an example for appeal to positive emotion: Your sibling is trying to convince you to let them eat the last piece of the dessert: “Can I have the last piece of cake? You know how much I love it, and it's been a tough day for me. I've had such a bad day, and this cake would just make me feel so much better.”
    Identify if this text contains the fallacy 'appeal to nature'. Here is an example for appeal to nature: Herbal medicine is natural, so it's good for you.
    Identify if this text contains the fallacy 'equivocation'. Here is an example for equivocation: Premise 1: Annoying co-workers are a headache. Premise 2: Painkillers can help you get rid of a headache. Conclusion: Painkillers can help you get rid of annoying co-workers.
    Identify if this text contains the fallacy 'appeal to anger'. Here is an example for appeal to anger: Are you tired of being ignored by your government?  Is it right that the top 1% have so much when the rest of us have so little?  I urge you to vote for me today!
    Identify if this text contains the fallacy 'false dilemma'. Here is an example for false dilemma: Either you support this new legislation to give the police more power, or you want society to descend into chaos!
    Identify if this text contains the fallacy 'straw man'. Here is an example for straw man: Person 1: I think we should increase benefits for unemployed single mothers during the first year after childbirth because they need sufficient money to provide medical care for their children. Person 2: So you believe we should give incentives to women to become single mothers and get a free ride from the tax money of hard-working citizens. This is just going to hurt our economy and our society in the long run.
    Identify if this text contains the fallacy 'appeal to tradition'. Here is an example for appeal to tradition: This medicine has been used by people since ancient history, therefore it must be an effective way to treat diseases.
    Identify if this text contains the fallacy 'appeal to pity'. Here is an example for appeal to tradition: You must have graded my exam incorrectly. I studied very hard for weeks specifically because I knew my career depended on getting a good grade. If you give me a failing grade I'm ruined!
    Identify if this text contains the fallacy 'tu quoque'. Here is an example for tu quoque: You used to do that when you were my age
    
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
