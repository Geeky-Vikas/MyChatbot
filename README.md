Hello eveyone, My name is vikas G, here this is my first chatbot using deep like pytorch and nltk with python 3

# My Chatbot using deep learning and Deployment with Flask and JavaScript

This gives 2 deployment options:
- Deploy within Flask app with jinja2 template
- Serve only the Flask prediction API. The used html and javascript files can be included in any Frontend application (with only a slight modification) and can run completely separate from the Flask App then.

## Initial Setup:
This repo currently contains the starter files.

Clone repo and create a virtual environment
```
$ gh repo clone Geeky-Vikas/MyChatbot
$ cd MyChatbot
$ python3 -m venv venv
$ . venv/bin/activate
```
Install dependencies
```
$ (venv) pip install Flask torch torchvision nltk
```
Install nltk package
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```
Modify `intents.json` with different intents and responses for your Chatbot

Run
```
$ (venv) python training.py
```
This will dump data.pth file. And then run
the following command to test it in the console.
```
$ (venv) python app.py
```

Now for deployment follow my tutorial to implement `app.py` and `app.js`.


## Note
In the video we implement the first approach using jinja2 templates within our Flask app. Only slight modifications are needed to run the frontend separately.

## Credits:
I  was inspired from frontend UI code:
https://github.com/hitchcliff/front-end-chatjs
