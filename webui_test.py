import gradio as gr
import run_demo
import argparse

# training params
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--channel_multiplier", type=int, default=1)
parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
parser.add_argument("--latent_dim_style", type=int, default=512)
parser.add_argument("--latent_dim_motion", type=int, default=20)
parser.add_argument("--source_path", type=str, default='')
parser.add_argument("--driving_path", type=str, default='')
parser.add_argument("--save_folder", type=str, default='./res')
args = parser.parse_args()

def run_model(imgpath, vidpath):
    parser.set_defaults(source_path=imgpath)
    parser.set_defaults(driving_path=vidpath)
    args = parser.parse_args()
    # print(args)
    demo = run_demo.Demo(args)
    demo.run()
    return "Finished!" #str(imgpath) + " --- " + str(vidpath)

with gr.Blocks() as demo:
    with gr.Row():
        img_source = gr.Image(label="Image for inpainting with mask", elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", visible=True, image_mode="RGBA")
        driving_video = gr.Video(type="filepath", label="Driving video")
    with gr.Row():
        submit_btn = gr.Button("Submit")
    with gr.Row():
        #output = gr.Video(label="Output")
        output = gr.Textbox(label="Output")
    submit_btn.click(fn=run_model, inputs=[img_source, driving_video], outputs=output)

demo.launch()