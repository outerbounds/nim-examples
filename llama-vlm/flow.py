from metaflow import FlowSpec, step, nim, current, card, IncludeFile, pypi
from metaflow.cards import Table, Image, Markdown

m = "meta/llama-3.2-11b-vision-instruct"
class HelloLlamaVLM(FlowSpec):

    terrace_chairs_bytes = IncludeFile(
        "img_file",
        is_text=False,
        help='Pass an image to demonstrate how to use Outerbounds, encode with base64, and pass to VLM.',
        default="uw-madison-terrace.jpeg"
    )

    @pypi(packages={"standard-imghdr": "3.13.0"})
    @card(id="results")
    @nim(models=[m])
    @step
    def start(self):
        import base64

        # Pass the IncludeFile result, or s3.get/get_many result to base64.
        # This is the way to handle images not on public internet.
        terrace_image_b64 = base64.b64encode(self.terrace_chairs_bytes).decode()
        terrace_image_data_uri = f"data:image/jpeg;base64,{terrace_image_b64}"

        self.openai_client_args = dict(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this images?"
                        },

                        ### OPTION 1: Public internet image URI 
                        # {
                        #     "type": "image_url",
                        #     "image_url":
                        #         {
                        #             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        #         }
                        # },

                        ### OPTION 2: Task runtime local image encoding
                        {
                            "type": "image_url",
                            "image_url":
                                {
                                    "url": terrace_image_data_uri
                                }
                        }
                        
                        ### Important note ###
                        # Currently, only one image per request is supported with Llama VLM NIMs.
                    ]
                }
            ],
            max_tokens=512
        )
        vlm = current.nim.models[m]
        resp = vlm(**self.openai_client_args)
        print(resp['choices'][0]['message']['content'])
        current.card["results"].append(Table(
            # headers=[],
            data=[[Image(self.terrace_chairs_bytes), Markdown(resp['choices'][0]['message']['content'])]]
        ))
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    HelloLlamaVLM()