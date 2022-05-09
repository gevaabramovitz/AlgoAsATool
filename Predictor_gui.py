import pickle

import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image,ImageTk
from tkinter.filedialog import askopenfile
import Predictor_engine
ERR_MSG_FILE_FORMAT = "Can't open this file"
ALREADY_LABELED_MSG = "the file is already labeled"


def open_file(browse_text, root):
    """
    this function handle with the file opening and prediction according to the
    trained model saved in the Predictor_engine.FILENAME_PICKLE_MET/MET and
    save the prediction in new file that contained the open file content with
    the predicted labels in addition
    :param browse_text: string var that wrote on the screen button
    :param root: Tk root
    :return: None
    """
    browse_text.set("loading...")
    file = askopenfile(parent=root, mode="rb", title="Choose a file", filetypes=[("Csv file","*.csv")])
    if file:
        #need to check the next line
        loaded_model_ket = pickle.load(open(Predictor_engine.FILENAME_PICKLE_KET, 'rb'))
        loaded_model_met = pickle.load(open(Predictor_engine.FILENAME_PICKLE_MET, 'rb'))
        X = pd.read_csv(file.name)
        X_out = X.copy()
        if("KET" in X.columns or "MET" in X.columns):
            return ALREADY_LABELED_MSG
        try:
            X_ket = Predictor_engine.load_data_make_features(X.copy(), True)
            X_met = Predictor_engine.load_data_make_features(X.copy(), False)

        except:
            return ERR_MSG_FILE_FORMAT
        y_predict_ket = loaded_model_ket.predict(X_ket)
        y_predict_met = loaded_model_met.predict(X_met)
        lamb1 = lambda x: True if x == 1 else False
        y_predict_ket = np.vectorize(lamb1)(np.array(y_predict_ket))
        y_predict_met = np.vectorize(lamb1)(np.array(y_predict_met))
        X_out["KET"] = y_predict_ket
        X_out["MET"] = y_predict_met
        X_out.to_csv("".join(file.name.split(".")[:-1]) + "_after_prediction.csv")
    browse_text.set("Browse")


def gui_runner():
    """
    this function runs the prediction gui of the met and ket diseases and
    activate the saved models for the prediction process
    :return: None
    """
    root = tk.Tk()
    # Set window title
    root.title("Cows' Diseases Predictor")
    canvas = tk.Canvas(root, width=600,height=300)
    canvas.grid(columnspan=3)
    #logo
    logo = Image.open("logo3.png")
    logo = ImageTk.PhotoImage(logo)
    logo_lable = tk.Label(image=logo)
    logo_lable.image = logo
    logo_lable.grid(column=1, row=0)
    #instruction
    instruction = tk.Label(root, text= "Select a cattle '.csv' file of cattle's data on your computer to predict", font="Raleway")
    instruction.grid(columnspan=3, column=0, row=1)
    #browser
    browse_text = tk.StringVar()
    browse_btn = tk.Button(root, textvariable=browse_text, command= lambda:open_file(browse_text,root), font="Raleway", bg="#80bebe", fg="white", height=2, width=15)
    browse_text.set("Browse")
    browse_btn.grid(column=1, row=2)
    root.mainloop()
if __name__ == '__main__':
    gui_runner()

