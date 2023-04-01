from tqdm import tqdm

prompt_template = """This task is to evaluate if the generated text contains hallucination given an input text. 
Hallucination is defined when there is information in the generated text that is not grounded in the input text or common sense, e.g. numbers, URLs or words with different meaning. 

Here are a few examples of hallucination:

Example 1
Input text:  "We sell Caribbean vacation cruises"
Generated text: "We sell tropical cruises for 1999$"

Example 2
Input text:  "Michelle Obama has a greate book self help on sale right now."
Generated text: "Check out Michelle Obama's great sel-help book here https://www.amazon.com/the-light-we-carry/dp/0593237463"

Example 3
Input text:  "Best paper towels"
Generated text: "Best paper towels made in the USA"

Example 4
Input text:  "Loose weight quick"
Generated text: "Lose 40lbs fast"

Note that ommiting information from the input text is not a hallucination.
Also, exaggerations that cannot be proven wrong, and that are often used in everyday’s language are not hallucination. Here are a few examples of non hallucination examples:

Example 1
Input text:  "We are a known bakery in town. We sell croissants, muffins, and cookies. They are all organically sourced and made fresh daily."
Generated text: "We are a known bakery in town. All our products are organically sourced."

Example 2
Input text:  "The best in town"
Generated text: "The greatest in the state"

Example 3
Input text:  "PLAY the #1 Bingo game on iPad for FREE!"
Generated text: "You can't win if you don't play the  Bingo game on iPad. PLAY NOW on iPad for FREE! 🤩 Play with friends or family!"

Given this task description, does below texts contain hallucination?
Input text: {input_text}
Generated text: {generated_text}

Please think step by step and answer "True" if it contains hallucination, or "False" if it does not.
"""


def complete(
    oai,
    df,
    model="gpt-3.5-turbo",
    input_text_column="body_1",
    generated_text_column="body_2",
    output_column="has-hallucination-gpt-3.5-turbo",
):
    print("using udpated CoT5")
    for index, row in tqdm(df.iterrows()):
        if output_column in row.index and (
            row[output_column] == "True"
            or row[output_column] == "False"
            or row[output_column] == "N/A"
        ):
            continue
        result = oai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt_template.format(
                        input_text=row[input_text_column],
                        generated_text=row[generated_text_column],
                    ),
                },
            ],
        )
        message = result["choices"][0]["message"]["content"]
        if "True" in message and "False" in message:
            df.loc[index, output_column] = "N/A"
        elif "True" in message:
            df.loc[index, output_column] = "True"
        elif "False" in message:
            df.loc[index, output_column] = "False"
        else:
            df.loc[index, output_column] = "N/A"
