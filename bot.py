from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from utils import SegmentDocument, GetDocument
from PIL import Image

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        text="Send a picture for prediction",
    )

def photo(update: Update, context: CallbackContext) -> None:
    bot = context.bot
    chat_id = update.message.chat_id
    photo_file = update.message.photo[-1].get_file()
    fimage = photo_file.download()
    image = Image.open(fimage)
    segment = SegmentDocument(image)
    # segmentation document image
    doc = segment.segmentImage()
    # float32 to uint8
    doc = segment.float32ToUint8(doc)

    find_doc = GetDocument(doc, image)
    # image convert PIL to numpy
    image = np.array(image)
    transformed = find_doc.transform(image)
    # RGB to BGR
    transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
    #open image file
    cv2.imwrite('predict_image.jpg', transformed)
    predict_image = open('predict_image.jpg', 'rb')
    
    # send image to telegram
    bot.send_photo(chat_id=chat_id, photo=predict_image)
    os.remove('predict_image.jpg')
    os.remove(fimage)
    

updater = Updater('1924852364:AAHQb7_xh2-umXrSsHi8s23py7f6aYNSIaE')


updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo))

updater.start_polling()
updater.idle()