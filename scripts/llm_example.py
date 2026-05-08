from vllm import LLM, SamplingParams

def generate_sample_responses():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is"
    ]

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )

    print("started to load the model")
    model_path = '/workspace/cs336/hf_cache/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots'
    model_hash = '4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2'
    full_path = model_path + '/' + model_hash
    print('model path: ' + full_path)
    llm = LLM(model=full_path)
    print("model loaded")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}m Generated text: {generated_text!r}')

if __name__ == '__main__':
    generate_sample_responses()