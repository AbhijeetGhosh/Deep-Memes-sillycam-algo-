from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask import url_for, current_app
from PIL import Image
import os
import secrets
from wtforms import SubmitField

#imports of first order model
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)


app.config['SECRET_KEY'] = 'abcdefghijklmnopqrstuvwxyz'

class UploadForm(FlaskForm):
	picture = FileField('Upload Image',validators=[FileAllowed(['jpg','png'])])
	submit = SubmitField('Submit')


def save_picture(form_picture):
	# random_hex = secrets.token_hex(8)
	# _, f_ext = os.path.splitext(form_picture.filename)
	# picture_fn = random_hex + f_ext
	# print(type(picture_fn))
	picture_path = os.path.join(current_app.root_path,'static/profile_pics', form_picture.filename) #saving image
	output_size = (500,500) #changing size
	i = Image.open(form_picture) #loading picture
	i.thumbnail(output_size)
	i.save(picture_path)
	generate_video(form_picture.filename)
	return form_picture.filename



@app.route('/',methods=['GET','POST'])
def home():
	form = UploadForm()
	if form.validate_on_submit():
		# 	
		print("Picture Uploaded")
		if form.picture.data:
			picture_file = save_picture(form.picture.data)
			image_file = url_for('static',filename='profile_pics/'+picture_file)	
			return render_template('home.html',title='Upload',image_file=image_file, form=form)
	# picture_file = url_for('static',filename='profile_pics/'+picture_file)
	return render_template('home.html',title='Upload', form=form)
	# return render_template('home.html',title='Upload',image_file=picture_file, form=form)


def generate_video(filename1):
	path1 = "static/profile_pics/" + filename1  #getting path
	source_image = imageio.imread(path1)  #loading image
	driving_video = imageio.mimread('t3.mp4')  #loading template

	# Resize image and video to 256x256

	source_image = resize(source_image, (256, 256))[..., :3]  #resizing image
	driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]  #resizing template

	# loading checkpoints
	print("loading checkpoints")
	from demo import load_checkpoints
	generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
											  checkpoint_path='vox-cpk.pth.tar')

	# performing image animation and saving video

	print("importing demo and skimage")
	from demo import make_animation
	from skimage import img_as_ubyte

	print("calling make_animation")
	predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

	# save resulting video
	# i_name , f_ext = os.path.splitext(filename1)
	# videoName = "static/generated_with_api_" + i_name + "mp4"
	imageio.mimsave("static/jahnavi_t4.mp4", [img_as_ubyte(frame) for frame in predictions])

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port = 8080)





