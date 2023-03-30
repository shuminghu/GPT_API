def complete(oai, df, model="gpt-3.5-turbo", input_column="body", output_column="gpt-3.5-turbo"):
    for index, row in df.iterrows():
        result = oai.ChatCompletion.create(
            model=model,
            messages=[
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": f"Please create 4 reformulations of the following sentence: {row[input_column]}"},
            ]
        )
        df.loc[index, output_column] = result['choices'][0]['message']['content']