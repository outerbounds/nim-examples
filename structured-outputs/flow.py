from metaflow import FlowSpec, step, nim, current

class NIMStructuredOutputs(FlowSpec):

    @nim(models=['meta/llama3-70b-instruct'])
    @step
    def start(self):
        import json 

        self.json_schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string"
                },
                "rating": {
                    "type": "number"
                }
            },
            "required": [
                "title",
                "rating"
            ]
        }
        self.system_prompt = (
            "You are a text extraction agent that reads text and outputs JSON."
            "Only output the title and the rating of the movie review in a valid JSON blob."
        )
        self.prompt = (
            f"Return the title and the rating based on the following movie review according to this JSON schema:{str(self.json_schema)}.\n"
            f"Review: Inception is a really well made film. I rate it four stars out of five."
        )

        llm = current.nim.models['meta/llama3-70b-instruct']
        self.response = llm(
            messages=[
                {"role": "assistant", "content": self.system_prompt},
                {"role": "user", "content": self.prompt}
            ],
            max_tokens=62,
            extra_body={"nvext": {"guided_json": self.json_schema}},
        )

        self.parsed_result = json.loads(self.response['choices'][0]['message']['content'])    
        assert self.parsed_result['title'] == "Inception"
        assert self.parsed_result['rating'] == 4
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    NIMStructuredOutputs()